# coding=utf-8
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
"""initialization."""

import random
import os
import time
import numpy as np

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed import initialize

from ascendspeed import get_adlr_autoresume
from ascendspeed import get_args
from ascendspeed import get_tensorboard_writer
from ascendspeed.core import tensor_parallel
from ascendspeed.utils import is_rank_0
from ascendspeed.core import parallel_state
from ascendspeed.global_vars import set_global_variables
from ascendspeed.core.parallel_state import (set_tensor_model_parallel_rank,
                                             set_tensor_model_parallel_world_size)
from ascendspeed.error_utils import check_equal


def initialize_megatron(extra_args_provider=None, args_defaults=None,
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using ascendspeed for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not args_defaults:
        args_defaults = {}

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert get_accelerator().is_available(), 'AscendSpeed requires accelerator.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed)

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Auto resume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None


def _compile_dependencies():
    if is_rank_0():
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        parallel_state,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)



def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    # 当前进程所在的node上可使用的GPU的数量
    device_count = get_accelerator().device_count()

    # 如果已创建好分布式环境
    if torch.distributed.is_initialized():
        # 在0号进程上打印出“创建完毕”的日志
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...',
                  flush=True)
        # 取得当前进程的全局序号
        args.rank = torch.distributed.get_rank()
        # 取得全局进程的个数
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        # 1. 初始化进程，分配GPU，并设置进程大组（group）
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                error_info = 'expected local-rank to be the same as rank % device-count.'
                check_equal(args.local_rank, device, error_info)
            else:
                args.local_rank = device

            get_accelerator().set_device(device)  # only do so when device_count > 0

        # Call the init process
        # 设置进程大组
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')  # 获取rank=0进程的ip
        master_port = os.getenv('MASTER_PORT', '6000')  # 获取rank=0进程的端口
        init_method += master_ip + ':' + master_port

        if args.deepspeed or args.ds_inference:
            deepspeed.init_distributed()
        else:
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size, rank=args.rank,
                init_method=init_method)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    # 2、制定DP/TP/PP分组策略，设置进程子组（subgroup）
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            parallel_state.initialize_model_parallel(args.tensor_model_parallel_size,
                                                     args.pipeline_model_parallel_size,
                                                     args.virtual_pipeline_model_parallel_size)

    # 设置DeepSpeed ZeRO-R，对activation进行优化
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _init_autoresume():
    """Set auto resume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        # No need to do so for CPU-only case.
        if get_accelerator().device_count() == 0:
            seed = seed_
        else:
            seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if get_accelerator().device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)),
                            global_step=args.iteration)


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        tensor_parallel.init_checkpointed_activations_memory_buffer()


def _is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or \
                torch.distributed.get_rank() % get_accelerator().device_count() == 0:
            return True
        else:
            return False
    else:
        return True


def adaptor_deepspeed_initialize(*largs, **kwargs):
    return_items = initialize(*largs, **kwargs)
    args = kwargs.get('args')
    if args is not None:
        if isinstance(return_items[0], PipelineEngine):
            return_items[0].is_pipe_partitioned = return_items[0].is_pipe_partitioned and not args.sequence_parallel
            return_items[0].is_grad_partitioned = return_items[0].is_grad_partitioned and not args.sequence_parallel

    return tuple(return_items)
