# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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
"""Pretrain Baichuan."""

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
from ascendspeed.core import parallel_state
from ascendspeed.core.enums import ModelType
from ascendspeed.data.gpt_dataset import build_train_valid_test_datasets
from ascendspeed.model.llama_model import LlamaModel, LlamaModelPipe
from ascendspeed.training import pretrain
from ascendspeed.utils import get_ltor_masks_and_position_ids
from ascendspeed.utils import average_losses_across_data_parallel_group
from ascendspeed.arguments import core_transformer_config_from_args
from pretrain_llama import (
    data_post_process,
    get_batch,
    get_batch_pipe,
    loss_func,
    forward_step,
    train_valid_test_datasets_provider
)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('Building Baichuan model ...')
    see_memory_usage(f"Before Building Model ...", force=True)

    args = get_args()
    config = core_transformer_config_from_args(get_args())
    with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=parallel_state):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = LlamaModelPipe(config, parallel_output=True)
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

if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=True)
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'PretrainedFromHF'},
             data_post_process=data_post_process)
