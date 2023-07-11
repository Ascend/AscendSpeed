# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from ascendspeed import get_args
from ascendspeed import print_rank_0
from ascendspeed import get_tokenizer
from ascendspeed import mpu
from ascendspeed.text_generation_utils import generate_samples_interactive
from ascendspeed.checkpointing import load_checkpoint
from ascendspeed.initialize import initialize_megatron
from ascendspeed.model.gpt_model import GPTModel
from ascendspeed.training import get_model
from ascendspeed.utils import get_ltor_masks_and_position_ids, unwrap_model
from ascendspeed.p2p_communication import recv_forward, send_forward
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.model import Float16Module
from deepspeed.accelerator import get_accelerator

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building bloom model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process, 
                     return_moe_loss=False)
    
    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser
	
def main():
    """Main program."""
    initialize_megatron(extra_args_provider=add_text_generate_args)

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Generate samples.
    assert args.num_samples == 0, "No sample is required for interactive inference"
    assert args.micro_batch_size == 1, "Interactive inference requires that micro_batch_size be set to 1"
    generate_samples_interactive(model)

if __name__ == "__main__":
    main()
