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

"""Pretrain InternLM"""
import logging

import torch
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

try:
    import torch_npu
    import deepspeed_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    logging.warning(f"Import torch_npu Error.")

from ascendspeed import get_args
from ascendspeed import print_rank_0
from ascendspeed.core import parallel_state
from ascendspeed.model.gpt_model import GPTModel, GPTModelPipe
from ascendspeed.core.enums import ModelType
from ascendspeed.training import pretrain
from pretrain_llama import (
    data_post_process,
    forward_step,
    train_valid_test_datasets_provider
)
from ascendspeed.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building InternLM model ...')
    see_memory_usage(f"Before Building Model", force=True)
    args = get_args()
    config = core_transformer_config_from_args(args)
    # internlm模型配置
    config.column_parallel_linear_bias = True
    config.row_parallel_linear_bias = True
    config.row_parallel_linear_skip_bias_add = False
    with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=parallel_state):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(config=config, parallel_output=True)
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
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
            model = GPTModel(
                config=config,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=True)
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'PretrainedFromHF'},
             data_post_process=data_post_process)
