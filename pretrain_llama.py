# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain Llama."""

import math
from functools import partial

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import deepspeed
import deepspeed_npu
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

from ascendspeed import get_args
from ascendspeed import print_rank_0
from ascendspeed import get_timers
from ascendspeed import get_tokenizer
from ascendspeed.core import parallel_state, tensor_parallel
from ascendspeed.data.gpt_dataset import build_train_valid_test_datasets as build_pretrain_dataset
from ascendspeed.data.decoder_packed_mtf_dataset import build_train_valid_test_datasets as build_instruction_dataset
from ascendspeed.model import LlamaModel, LlamaModelPipe
from ascendspeed.core.enums import ModelType
from ascendspeed.training import pretrain
from ascendspeed.utils import get_ltor_masks_and_position_ids, get_tune_attention_mask
from ascendspeed.utils import average_losses_across_data_parallel_group
from ascendspeed.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('Building llama model ...')
    see_memory_usage(f"Before Building Model ...", force=True)

    args = get_args()
    config = core_transformer_config_from_args(get_args())
    with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=parallel_state):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = LlamaModelPipe(config=config, parallel_output=True)
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length),
                device=get_accelerator().current_device_name())).view(
                1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)
        else:
            model = LlamaModel(
                config=config,
                parallel_output=True,
                add_pooler=False,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    # Broadcast data.
    if hasattr(data_iterator, '__next__'):
        data = next(data_iterator)
    else:
        if isinstance(data_iterator, list):
            return data_iterator.pop(0)
        else:
            data = None

    (tokens, attention_mask), (labels, loss_mask) = get_batch_pipe(data)
    args = get_args()
    if args.foldx_mode is not None:
        if hasattr(data_iterator, 'dummy_iterators'):
            for iterator in data_iterator.dummy_iterators:
                iterator.append((tokens, labels, loss_mask, attention_mask,))

    return tokens, labels, loss_mask, attention_mask


def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen + 1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen + 1)) * (current_seqlen + 1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen + 1),
                                          data['text'][:, -(current_seqlen + 1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen + 1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    if args.is_instruction_dataset:
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        data_type = torch.int64

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, data, data_type)

        # Unpack.
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == -100, 0, 1)

        attention_mask = get_tune_attention_mask(attention_mask_1d, args.reset_attention_mask)

        return (tokens, attention_mask), (labels, loss_mask)

    # Items and their type.
    keys = ['text']
    data_type = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, data_type)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, _ = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    return (tokens, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()

    timers = get_timers()
    # Get the batch.
    if args.foldx_mode is None:
        timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask = get_batch(data_iterator)
    if args.foldx_mode is None:
        timers('batch-generator').stop()

    output_tensor = model(tokens, attention_mask, labels=labels)
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for llama ...')
    build_dataset_func = build_instruction_dataset if args.is_instruction_dataset else build_pretrain_dataset
    train_ds, valid_ds, test_ds = build_dataset_func(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating llama datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=True)
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'PretrainedFromHF'},
             data_post_process=data_post_process)
