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
"""Pretrain utilities."""
from datetime import datetime
import math
import sys
import time
import json

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd

from ascendspeed import get_args
from ascendspeed import get_timers
from ascendspeed import get_tensorboard_writer
from ascendspeed import get_current_global_batch_size
from ascendspeed import get_num_microbatches
from ascendspeed import is_last_rank
from ascendspeed import update_num_microbatches
from ascendspeed import print_rank_0
from ascendspeed import print_rank_last
from ascendspeed.core import tensor_parallel, parallel_state
from ascendspeed.checkpointing import load_checkpoint
from ascendspeed.checkpointing import save_checkpoint
from ascendspeed.model import Float16Module
from ascendspeed.core.enums import ModelType
from ascendspeed.optimizer import get_megatron_optimizer
from ascendspeed.initialize import initialize_megatron
from ascendspeed.initialize import write_args_to_tensorboard
from ascendspeed.learning_rates import AnnealingLR
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.utils import check_adlr_autoresume_termination
from ascendspeed.utils import unwrap_model
from ascendspeed.data.data_samplers import build_pretraining_data_loader
from ascendspeed.utils import calc_params_l2_norm
from ascendspeed.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator
from ascendspeed.model.transformer import ParallelTransformerLayer
from ascendspeed.model.lora_utils import is_enable_lora, handle_model_with_lora
from ascendspeed.core.pipeline_parallel.schedules import forward_backward_pipelining_with_foldx_fifo
from ascendspeed.core.pipeline_parallel.schedules import forward_backward_pipelining_with_foldx_aiao
from ascendspeed.core.pipeline_parallel.schedules import get_forward_backward_func, get_forward_func
from ascendspeed.error_utils import check_equal, check_type, ensure_var_is_not_none, ensure_var_is_none
from ascendspeed.core.utils import get_model_config
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def _initialize_optimized_pipeline():
    args = get_args()
    if args.manual_mbs == 'example-config-1':
        # An example config when pipeline-model-parallel-size is 4.
        # This theoretically reduces near 20% pipeline bubble.
        check_equal(args.micro_batch_size, 4)
        check_equal(args.global_batch_size // parallel_state.get_data_parallel_world_size(), 64)
        check_equal(args.pipeline_model_parallel_size, 4)
        args.manual_mbs = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, \
                           5, 5, 5, 4, 3, 2]
    elif args.manual_mbs == 'example-config-2':
        # An example config when pipeline-model-parallel-size is 8
        # # This theoretically reduces near 30% pipeline bubble.
        check_equal(args.micro_batch_size, 4)
        check_equal(args.global_batch_size // parallel_state.get_data_parallel_world_size(), 96)
        check_equal(args.pipeline_model_parallel_size, 8)
        args.manual_mbs = [1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, \
                           5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 3, 2]
    elif args.maual_mbs is not '':
        # Customized manual micro-batch-size
        # Warning: this API will be changed in the future
        # to automatically set args.maual_mbs for minimizing
        # bubble time in pipeline.
        mbs = args.manual_mbs.split(',')
        mbs = [int(mbs[i]) for i in range(len(mbs))]
        args.manual_mbs = mbs
    else:
        raise ValueError('A proper manual-mbs has to be provided.')

    # sanity check
    error_message = 'A proper manual-mbs has to be provided'
    check_type(args.manual_mbs, list, error_message)
    
    assert len(args.manual_mbs) == args.global_batch_size // parallel_state.get_data_parallel_world_size() \
           // args.micro_batch_size, 'Check number of micro batches.'
    assert sum(args.manual_mbs) * parallel_state.get_data_parallel_world_size() == args.global_batch_size, \
        'Check either miro batch sizes or global batch sizes.'


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize ascendspeed.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla, we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g. images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initialize and get arguments, timers, and TensorBoard writer.
    # 1.初始化分布式环境
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time, so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = get_accelerator().FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize ascendspeed (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after ascendspeed is initialized.')

    args = get_args()
    timers = get_timers()

    if args.optimized_pipeline:
        _initialize_optimized_pipeline()

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        if "curriculum_learning" in args.deepspeed_configuration and \
                "enabled" in args.deepspeed_configuration["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_configuration[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_configuration["curriculum_learning"])
        if "compression_training" in args.deepspeed_configuration:
            args.compression_training = True

    # Model, optimizer, and learning rate.
    # 2、模型并行：定义模型架构，并切割模型
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])
    # Data stuff.
    # 3、构造train/val/test数据集
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
        if args.foldx_mode is not None:
            train_data_iterator = [[] for _ in all_data_iterators]
            if all_data_iterators[0][0] is None:
                from types import SimpleNamespace
                train_data_iterator[0] = SimpleNamespace()
            else:
                train_data_iterator[0] = all_data_iterators[0][0]
            train_data_iterator[0].dummy_iterators = train_data_iterator[1:]
        valid_data_iterator = [[
            all_data_iterators[i][1][j] for i in range(len(all_data_iterators))]
            for j in range(len(all_data_iterators[0][1]))
        ]
        test_data_iterator = [[
            all_data_iterators[i][2][j] for i in range(len(all_data_iterators))]
            for j in range(len(all_data_iterators[0][2]))
        ]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    if args.data_efficiency_curriculum_learning:
        if args.deepspeed_dataloader is not None:
            # We use args to pass the deepspeed_dataloader because adding
            # output to setup_model_and_optimizer will break the API for other
            # cases. We clear args.deepspeed_dataloader after updating
            # train_data_iterator because args will be saved in checkpoint and
            # attempting to save the whole deepspeed_dataloader will lead to
            # "AttributeError: Can't pickle local object...".
            train_data_iterator = iter(args.deepspeed_dataloader)
            args.deepspeed_dataloader = None
        else:
            train_data_iterator = None
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model().
    args.teacher_model = None
    if args.mos or args.kd:  # Set up teacher model.
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)
    print_rank_0('training ...')

    # 4、正式训练
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator, config)
    print_datetime('after training is done')

    if args.do_valid:
        prefix = 'the end of training for val data'
        for iterator in valid_data_iterator:
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       iteration, False)

    # Clean the model and do evaluation again
    if args.compression_training:
        model = [redundancy_clean(model[0], args.deepspeed_config, tensor_parallel)]
        if args.do_valid:
            prefix = 'the end of training and after model cleaning for val data'
            for iterator in valid_data_iterator:
                evaluate_and_print_results(prefix, forward_step_func,
                                           iterator, model,
                                           iteration, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        for iterator in test_data_iterator:
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       0, True)


def update_train_iters(args):
    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def setup_teacher_model(args, model_provider):
    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    print_rank_0('***>>>>> Setting up the teacher model')

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type
    # Build model.
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type

            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                ensure_var_is_not_none(args.pipeline_model_parallel_split_rank, error_message="Split rank needs"\
                                       " to be specified for model with both encoder and decoder")
                rank = parallel_state.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = parallel_state.is_pipeline_stage_before_split()
                add_decoder = parallel_state.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    if is_enable_lora():
        model = handle_model_with_lora(model)

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if parallel_state.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p, 'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if not args.deepspeed:
        # GPU allocation.
        for model_module in model:
            device_name = get_accelerator().current_device_name()
            print_rank_0(f"model to {device_name}")
            model_module.to(device_name)

        model = wrap_model(model, wrap_with_ddp=wrap_with_ddp)

    return model


def wrap_model(model, wrap_with_ddp=True):
    args = get_args()
    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]
    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = get_accelerator().current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=parallel_state.get_data_parallel_group())
                     for model_module in model]
            return model

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            return model
        else:
            raise NotImplementedError('Unknown DDP implementation specified: {}. '
                                      'Exiting.'.format(args.DDP_impl))

    return model


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        with open(args.deepspeed_config, 'r') as fd:
            ds_config = json.load(fd)

        # When loading just the model weights, ZeRO can be disabled.
        if 'zero_optimization' in ds_config:
            del ds_config['zero_optimization']

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            config=ds_config
        )

        assert not isinstance(model, deepspeed.PipelineEngine), \
            'Weight loading only mode is not supported in pipeline parallelism yet.'

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)
    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              teacher=False,
                              data_post_process=None,
                              build_train_valid_test_datasets_provider=None):
    """Setup model and optimizer."""
    args = get_args()
    model = get_model(model_provider_func, model_type)
    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=parallel_state if args.no_pipeline_parallel else None
        )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        print_rank_0('***>>>>> Student model, global step:{}'.format(student_global_steps))

    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=parallel_state if args.no_pipeline_parallel else None
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config, tensor_parallel)]

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        lr_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:
            optimizer = get_megatron_optimizer(model)
        lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = parallel_state.get_pipeline_model_parallel_world_size()
        if args.data_efficiency_curriculum_learning and build_train_valid_test_datasets_provider is not None:
            train_ds = None
            # Only need to build dataset on tp rank 0 since ascendspeed has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                    update_train_iters(args)
                else:
                    train_samples = args.train_iters * args.global_batch_size
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (args.train_iters // args.eval_interval + 1) * \
                             args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [train_samples,
                                              eval_iters * args.global_batch_size,
                                              test_iters * args.global_batch_size]
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples)
            model, optimizer, args.deepspeed_dataloader, lr_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                training_data=train_ds,
                mpu=parallel_state if args.no_pipeline_parallel else None
            )
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=lr_scheduler,
                mpu=parallel_state if args.no_pipeline_parallel else None
            )
            check_equal(model.fp16_enabled(), args.fp16, error_info="megatron fp16 config does not match deepspeed")
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            check_equal(model.grid.get_pipe_parallel_rank(), parallel_state.get_pipeline_model_parallel_rank())
            check_equal(model.grid.get_slice_parallel_rank(), parallel_state.get_tensor_model_parallel_rank())
            check_equal(model.grid.get_data_parallel_rank(), parallel_state.get_data_parallel_rank())
        model = [model]

    # Compression has its own checkpoint loading path (e.g, loading both teacher and student models). So if compression is enabled, we skip the following checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        print_rank_0(f"\tsetup_model_and_optimizer : no_post_init_checkpoint_loading:{no_post_init_checkpoint_loading}")
        if args.load is not None:
            timers = get_timers()
            print_rank_0(f"\tsetup_model_and_optimizer : args.load:{args.load}")
            # Extra barrier is added to make sure all ranks report the
            # max time.
            torch.distributed.barrier()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer.reload_model_params()
            torch.distributed.barrier()
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or parallel_state.get_pipeline_model_parallel_world_size() > 1:
        check_equal(args.DDP_impl, 'local')

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        check_type(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        grad_norm = model[0].get_global_grad_norm()
        return {'lm loss': loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        else:
            optimizer.zero_grad()

    timers('forward-backward', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    forward_backward_func = get_forward_backward_func()

    if args.mos or args.kd:
        # args.teacher_forward is used as global variable to enable kd loss
        # calculation in forward pass. Users do not need to set it in the
        # command line to use kd.
        args.teacher_forward = True
    if forward_backward_func == forward_backward_pipelining_with_foldx_fifo or\
            forward_backward_func == forward_backward_pipelining_with_foldx_aiao:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length)
    else:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    if args.mos or args.kd:
        args.teacher_forward = False
    # reset timers if necessary
    if config.timers is None:
        config.timers = timers
    timers('forward-backward').stop()

    # All-reduce if needed.
    if not args.deepspeed and args.DDP_impl == 'local':
        timers('backward-params-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
        if args.foldx_mode is not None:
            handles = model[0].allreduce_gradients(async_op=True)
            for handle in handles:
                handle.wait()
        else:
            for model_module in model:
                model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce', log_level=1).start(barrier=args.barrier_with_L1_time)
    if not args.deepspeed:
        optimizer.reduce_model_grads(args, timers)
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'increment': increment})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
        if update_successful:
            optimizer.gather_model_params(args, timers)
    timers('optimizer').stop()

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None

        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers._timers:
            timers_to_log.append(name)

    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-backward-send-forward-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-send-backward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')
    add_to_logging('save-checkpoint')

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
                 get_num_microbatches()
    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0) and \
            is_last_rank():
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                              args.consumed_train_tokens)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if hasattr(args, 'actual_seq_length'):
            writer.add_scalar('seqlen/actual_seq_length', args.actual_seq_length,
                              iteration)
            writer.add_scalar('seqlen/actual_seq_length vs samples', args.actual_seq_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/actual_seq_length vs tokens', args.actual_seq_length,
                              args.consumed_train_tokens)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            writer.add_scalar('seqlen/curriculum_seqlen', args.curriculum_seqlen,
                              iteration)
            writer.add_scalar('seqlen/curriculum_seqlen vs samples', args.curriculum_seqlen,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/curriculum_seqlen vs tokens', args.curriculum_seqlen,
                              args.consumed_train_tokens)
        if args.random_ltd:
            writer.add_scalar('seqlen/random_ltd_reserved_length', args.random_ltd_reserved_length,
                              iteration)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs samples', args.random_ltd_reserved_length,
                              args.consumed_train_samples)
            writer.add_scalar('seqlen/random_ltd_reserved_length vs tokens', args.random_ltd_reserved_length,
                              args.consumed_train_tokens)
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)

    if iteration % args.tensorboard_log_interval == 0:
        # This logging write various optimizer states to tensorboard. This
        # feature may consume extra GPU memory thus is set at false by default.
        if args.log_optimizer_states_to_tensorboard and optimizer is not None:
            opt_stats = [0.0] * 8
            opt_stats_2 = [0.0] * 4
            for _, group in enumerate(optimizer.param_groups):
                for _, param in enumerate(group['params']):
                    opt_stats[0] += (torch.norm(optimizer.state[param]['exp_avg_sq']).item()) ** 2
                    opt_stats[1] += (torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt()).item()) ** 2
                    opt_stats[2] += (torch.norm(optimizer.state[param]['exp_avg']).item()) ** 2
                    opt_stats[3] += (torch.norm(param).item()) ** 2
                    opt_stats[4] += torch.norm(optimizer.state[param]['exp_avg_sq'], p=1).item()
                    opt_stats[5] += torch.norm(optimizer.state[param]['exp_avg_sq'].sqrt(), p=1).item()
                    opt_stats[6] += torch.norm(optimizer.state[param]['exp_avg'], p=1).item()
                    opt_stats[7] += torch.norm(param, p=1).item()
                    opt_stats_2[0] = max(opt_stats_2[0], abs(optimizer.state[param]['exp_avg_sq'].max().item()),
                                         abs(optimizer.state[param]['exp_avg_sq'].min().item()))
                    opt_stats_2[1] = max(opt_stats_2[1], optimizer.state[param]['exp_avg_sq']
                                         .sqrt().abs_().max().item())
                    opt_stats_2[2] = max(opt_stats_2[2], abs(optimizer.state[param]['exp_avg'].max().item()),
                                         abs(optimizer.state[param]['exp_avg'].min().item()))
                    opt_stats_2[3] = max(opt_stats_2[3], abs(param.max().item()), abs(param.min().item()))

            if args.zero_stage > 0:
                # ZeRO partiions optimizer states
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=parallel_state.get_data_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                                             group=parallel_state.get_data_parallel_group())

            if args.tensor_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=parallel_state.get_tensor_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                                             group=parallel_state.get_tensor_model_parallel_group())

            if args.pipeline_model_parallel_size > 1:
                opt_stats = get_accelerator().FloatTensor(opt_stats)
                torch.distributed.all_reduce(opt_stats, group=parallel_state.get_pipeline_model_parallel_group())
                opt_stats_2 = get_accelerator().FloatTensor(opt_stats_2)
                torch.distributed.all_reduce(opt_stats_2, op=torch.distributed.ReduceOp.MAX,
                                             group=parallel_state.get_pipeline_model_parallel_group())

            # print('step {} rank {} after sync opt_stats {}, {}'.format(iteration, torch.distributed.get_rank(), opt_stats_2, opt_stats))
            if writer and is_last_rank():
                writer.add_scalar('optimizer/variance_l2 vs tokens', opt_stats[0] ** 0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l2 vs tokens', opt_stats[1] ** 0.5,
                                  args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l2 vs tokens', opt_stats[2] ** 0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l2 vs tokens', opt_stats[3] ** 0.5, args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_l1 vs tokens', opt_stats[4], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_l1 vs tokens', opt_stats[5], args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_l1 vs tokens', opt_stats[6], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_l1 vs tokens', opt_stats[7], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_abs_max vs tokens', opt_stats_2[0], args.consumed_train_tokens)
                writer.add_scalar('optimizer/variance_sqrt_abs_max vs tokens', opt_stats_2[1],
                                  args.consumed_train_tokens)
                writer.add_scalar('optimizer/momentum_abs_max vs tokens', opt_stats_2[2], args.consumed_train_tokens)
                writer.add_scalar('optimizer/weight_abs_max vs tokens', opt_stats_2[3], args.consumed_train_tokens)

                writer.add_scalar('optimizer/variance_l2', opt_stats[0] ** 0.5, iteration)
                writer.add_scalar('optimizer/variance_sqrt_l2', opt_stats[1] ** 0.5, iteration)
                writer.add_scalar('optimizer/momentum_l2', opt_stats[2] ** 0.5, iteration)
                writer.add_scalar('optimizer/weight_l2', opt_stats[3] ** 0.5, iteration)
                writer.add_scalar('optimizer/variance_l1', opt_stats[4], iteration)
                writer.add_scalar('optimizer/variance_sqrt_l1', opt_stats[5], iteration)
                writer.add_scalar('optimizer/momentum_l1', opt_stats[6], iteration)
                writer.add_scalar('optimizer/weight_l1', opt_stats[7], iteration)
                writer.add_scalar('optimizer/variance_abs_max', opt_stats_2[0], iteration)
                writer.add_scalar('optimizer/variance_sqrt_abs_max', opt_stats_2[1], iteration)
                writer.add_scalar('optimizer/momentum_abs_max', opt_stats_2[2], iteration)
                writer.add_scalar('optimizer/weight_abs_max', opt_stats_2[3], iteration)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        vocab_size = args.padded_vocab_size

        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(model, args, elapsed_time,
                                                                                       total_iterations)

        # Compute throughput.
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size

        # only the last rank process has a non-None _GLOBAL_TENSORBOARD_WRITER
        if writer and is_last_rank():
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = get_accelerator().FloatTensor([0.0])
        if loss_scale is not None:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        if args.random_ltd:
            log_string += ' random ltd reserved length: {:5d} |'.format(args.random_ltd_reserved_length)
        log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save-checkpoint').stop(barrier=True)
    checkpoint_throughput_calculator(model, timers('save-checkpoint').elapsed(reset=False))
    timers.log(['save-checkpoint'])


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    if args.random_ltd:
        # random-ltd requires different randomness on each rank
        import random
        random.seed(args.seed + torch.distributed.get_rank())

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Translate args to core configuration
    if not args.deepspeed:
        config.grad_scale_func = optimizer.scale_loss
    config.timers = timers

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    if args.random_ltd:
        assert model[0].random_ltd_enabled()
        args.random_ltd_layer_num = model[0].random_ltd_scheduler.get_random_ltd_layer_num()

    while iteration < args.train_iters and (args.train_tokens is None or \
                                            args.consumed_train_tokens < args.train_tokens):
        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = parallel_state.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)

        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                args.iteration + 1)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       lr_scheduler, config)
        iteration += 1
        args.iteration = iteration
        new_samples = parallel_state.get_data_parallel_world_size() * \
                      args.micro_batch_size * \
                      get_num_microbatches()
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[0].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (args.actual_seq_length * (
                        args.num_layers - args.random_ltd_layer_num)
                        + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += parallel_state.get_data_parallel_world_size() * \
                                              get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length

        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          model, optimizer)

        # Autoresume
        if args.adlr_autoresume and \
                (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
                args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            for iterator in valid_data_iterator:
                evaluate_and_print_results(prefix, forward_step_func,
                                           iterator, model,
                                           iteration, False)

        # Checkpointing
        saved_checkpoint = False
        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     lr_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = get_accelerator().IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         lr_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            model[0].reset_activation_shape()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
                            (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))
            forward_backward_func = get_forward_func()
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss': loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)

            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if 'moe' not in key:
                            total_loss_dict[key] = total_loss_dict.get(
                                key, get_accelerator().FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += parallel_state.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            model[0].reset_activation_shape()

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False, test=False, **kwargs):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    ds_name = kwargs.get("data_group_name", None)
    # print corresponding dataset name (used for multiple validation datasets)
    lm_loss_validation = f"lm-loss-validation/{ds_name}" if ds_name else "lm-loss-validation"

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            data_type = 'test' if test else 'validation'
            writer.add_scalar(f'{lm_loss_validation}/{key} {data_type}',
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar(f'{lm_loss_validation}/{key} {data_type} vs samples',
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar(f'{lm_loss_validation}/{key} {data_type} vs tokens',
                              total_loss_dict[key].item(),
                              args.consumed_train_tokens)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'{lm_loss_validation}/{key} {data_type} ppl', ppl,
                                  iteration)
                writer.add_scalar(f'{lm_loss_validation}/{key} {data_type} ppl vs samples',
                                  ppl, args.consumed_train_samples)
                writer.add_scalar(f'{lm_loss_validation}/{key} {data_type} ppl vs tokens',
                                  ppl, args.consumed_train_tokens)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloaders, valid_dataloaders, test_dataloaders) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        ensure_var_is_none(args.train_samples, error_message='only backward compatiblity'\
                           ' support for iteration-based training')
        args.consumed_train_samples = args.iteration * args.global_batch_size

    if args.iteration // args.eval_interval > 0 and args.consumed_valid_samples == 0:
        ensure_var_is_none(args.train_samples, error_message='only backward compatiblity'\
                           ' support for iteration-based training')
        args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                                      args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if parallel_state.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
            update_train_iters(args)
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)

        # if dataloading option is not 2 convert to list to allow
        # same interface for multiple data groups
        # for validation and testing in option 2
        if type(train_ds) != list and train_ds is not None:
            train_ds = [train_ds]
        if type(valid_ds) != list and valid_ds is not None:
            valid_ds = [valid_ds]
        if type(test_ds) != list and test_ds is not None:
            test_ds = [test_ds]

        # Build dataloders.
        train_dataloaders = build_pretraining_data_loader(train_ds[0], args.consumed_train_samples)

        valid_dataloaders = [build_pretraining_data_loader(d, args.consumed_valid_samples // len(valid_ds))
                             for d in valid_ds] \
            if valid_ds is not None else []

        # We collapse None and empty list as both should mean we don't run test
        test_dataloaders = [build_pretraining_data_loader(d, 0) for d in test_ds] \
            if test_ds is not None else []

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloaders is not None and args.train_iters > 0
        do_valid = valid_dataloaders is not None and args.eval_iters > 0
        do_test = test_dataloaders is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = get_accelerator().LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = get_accelerator().LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                parallel_state.get_tensor_model_parallel_src_rank(),
                                group=parallel_state.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    num_valid_ds = flags[1].item()
    num_test_ds = flags[2].item()
    assert num_test_ds >= 0
    assert num_valid_ds >= 0
    args.do_valid = num_valid_ds > 0
    args.do_test = num_test_ds > 0

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloaders is not None:
        train_data_iterators = iter(train_dataloaders) if dl_type == 'single' \
            else iter(cyclic_iter(train_dataloaders))
    else:
        train_data_iterators = None

    if valid_dataloaders is not None:
        valid_data_iterators = [iter(vdl) if dl_type in ['single'] \
                                    else iter(cyclic_iter(valid_dataloaders))
                                for vdl in valid_dataloaders]
    else:
        valid_data_iterators = [None] * num_valid_ds

    if test_dataloaders is not None:
        test_data_iterators = [iter(tdl) if dl_type in ['single'] \
                                   else iter(cyclic_iter(test_dataloaders))
                               for tdl in test_dataloaders]
    else:
        test_data_iterators = [None] * num_test_ds

    return train_data_iterators, valid_data_iterators, test_data_iterators
