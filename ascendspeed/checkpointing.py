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

"""Input/output checkpointing."""

import os
import random
import sys
import traceback

import numpy as np
from deepspeed import PipelineEngine
from deepspeed.accelerator import get_accelerator
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from ascendspeed.enums import PositionEmbeddingType
from ascendspeed.utils import WRITE_FILE_DEFAULT_FLAGS, WRITE_FILE_DEFAULT_MODES

from ascendspeed import (get_args,
                         is_rank_0,
                         print_rank_0,
                         update_num_microbatches,
                         utils)
from ascendspeed.core import parallel_state, tensor_parallel
from ascendspeed.model import DistributedDataParallel as LocalDDP, Float16Module
from ascendspeed.model.lora_utils import is_enable_lora, get_lora_state_dict, lora_custom_load_fn_for_deepspeed, \
    get_lora_model_classes, get_lora_state_dict_with_deepspeed, update_model_state_dict_with_megatron, \
    get_lora_load_fn_with_deepspeed, handle_lora_modules_to_save_key_with_megatron
from ascendspeed.error_utils import check_equal, ensure_valid

_CHECKPOINT_VERSION = None


def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        error_info = "checkpoint versions do not match"
        check_equal(_CHECKPOINT_VERSION, value, error_info)
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def check_checkpoint_args(checkpoint_args):
    """
    Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint.
    """
    args = get_args()

    def _compare(arg_name, old_arg_name=None):
        if old_arg_name is not None:
            checkpoint_value = getattr(checkpoint_args, old_arg_name)
        else:
            checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_info = '{} value from checkpoint ({}) is not equal to the ' \
                     'input argument value ({}).'.format(
            arg_name, checkpoint_value, args_value)
        check_equal(checkpoint_value, args_value, error_info)

    if not args.mos and not args.kd:
        _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('position_embedding_type')
    # with alibi we can change `max_position_embeddings`
    if args.position_embedding_type != PositionEmbeddingType.alibi:
        _compare('max_position_embeddings')

    if args.vocab_file:
        _compare('make_vocab_size_divisible_by')
        _compare('padded_vocab_size')
        _compare('tokenizer_type')
    if get_checkpoint_version() < 3.0:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0:
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration,
                        release=False, model_name='model_optim_rng.pt'):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                parallel_state.get_tensor_model_parallel_rank()),
                            model_name)
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            parallel_state.get_tensor_model_parallel_rank(),
                            parallel_state.get_pipeline_model_parallel_rank()),
                        model_name)


def get_checkpoint_tracker_filename(checkpoints_path):
    """
    Tracker file rescords the latest chckpoint during
    training to restart from.
    """
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    if not args.deepspeed:
        unwrap_model_classes = (torchDDP, LocalDDP, Float16Module)
        if is_enable_lora():
            unwrap_model_classes += get_lora_model_classes()
        model = utils.unwrap_model(model, unwrap_model_classes)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    if not torch.distributed.is_initialized() or parallel_state.get_data_parallel_rank() == 0 \
            or args.deepspeed:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        state_dict['tokens'] = args.consumed_train_tokens

        # DeepSpeed saves the model/optimizer/scheduler
        if not args.deepspeed:
            get_model_state_dict(model, state_dict)

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    state_dict['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
            state_dict['cuda_rng_state'] = get_accelerator().get_rng_state()
            state_dict['rng_tracker_states'] \
                = tensor_parallel.get_cuda_rng_tracker().get_states()

        # Save.
        checkpoint_name = get_checkpoint_name(args.save, iteration)
        if not args.deepspeed:
            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)

    if args.deepspeed:
        original_state_dict = None
        # ascendspeed model uses state_dict_for_save_checkpointing instead of the standard state_dict
        # state_dict is used by deepspeed for module saving so it needs to point to the right function
        if args.no_pipeline_parallel:
            original_state_dict = model[0].module.state_dict

            def state_dict_for_save_checkpoint_deepspeed(destination=None, prefix='', keep_vars=False):
                return model[0].module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

            model[0].module.state_dict = state_dict_for_save_checkpoint_deepspeed
        if is_enable_lora():
            if original_state_dict is None:
                original_state_dict = model[0].module.state_dict
            model[0].module.state_dict = get_lora_state_dict_with_deepspeed(model=model[0])

        # Saving is a collective communication
        checkpoint_name = get_checkpoint_name(args.save, iteration)

        # Trim off the filename and mp_rank_* directory.
        for _ in range(3):
            checkpoint_name = os.path.dirname(checkpoint_name)
        model[0].save_checkpoint(checkpoint_name, client_state=state_dict)

        if original_state_dict is not None:
            model[0].module.state_dict = original_state_dict

    save_checkpoint_post_process(iteration)


def get_model_state_dict(model, state_dict):
    if len(model) == 1:
        state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        if is_enable_lora():
            state_dict['model'] = get_lora_state_dict(state_dict['model'])
    else:
        for i in range(len(model)):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()
            if is_enable_lora():
                state_dict['model%d' % i] = get_lora_state_dict(state_dict['model%d' % i])


def save_checkpoint_post_process(iteration):
    args = get_args()

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # And update the latest iteration
    if is_rank_0():
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with os.fdopen(os.open(tracker_filename, WRITE_FILE_DEFAULT_FLAGS, WRITE_FILE_DEFAULT_MODES), 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, 'module'):
        model = model.module
    # attention_module = model.language_model.encoder.layers[0].self_attention
    attention_module = model.language_model.encoder.layers[0].attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_splits, num_attention_heads_per_partition,
             hidden_size_per_attention_head) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h] """

        intermediate_shape = \
            (num_attention_heads_per_partition,
             hidden_size_per_attention_head, num_splits) + \
            input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            check_equal(len(model), 1)
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith(('.query_key_value.weight', '.query_key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith(('.key_value.weight', '.key_value.bias')):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(" succesfully fixed query-key-values ordering for"
                     " checkpoint version {}".format(checkpoint_version))


def read_tracker(load_dir):
    args = get_args()
    iteration = 0
    release = False
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iteration zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return False, iteration, release

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()

    if not args.mos and not args.kd:
        error_message = 'error parsing metadata file {}'.format(tracker_filename)
        ensure_valid(iteration > 0 or release, error_message)

    return True, iteration, release


def get_state_dict_and_release(load_dir, lora_load_dir=None):
    args = get_args()

    read_tracker_success, iteration, release = read_tracker(load_dir)
    if not read_tracker_success:
        raise ValueError(f"{load_dir} do not have tracker.")
    if lora_load_dir:
        read_tracker_success, lora_iteration, lora_release = read_tracker(lora_load_dir)
        if not read_tracker_success:
            raise ValueError(f"{lora_load_dir} do not have tracker.")

    # Checkpoint.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    print_rank_0(f' loading checkpoint from {args.load} at iteration {iteration}')
    model_checkpoint_name = None
    if lora_load_dir:  # 有lora目录时，其他参数都应从lora目录读取，load目录只提供原始模型权重
        model_checkpoint_name = checkpoint_name
        checkpoint_name = get_checkpoint_name(lora_load_dir, lora_iteration, lora_release)
        print_rank_0(
            f' loading lora checkpoint from {args.lora_load} at iteration {lora_iteration} release:{lora_release}')
        release = lora_release

    # Load the checkpoint.
    try:
        state_dict = load_state_dict_from_checkpoint_with_megatron(checkpoint_name,
                                                                   model_checkpoint_name=model_checkpoint_name)
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler
        # For backward compatibility.
        print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        state_dict = load_state_dict_from_checkpoint_with_megatron(checkpoint_name,
                                                                   model_checkpoint_name=model_checkpoint_name)
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        traceback.print_exc()
        sys.exit()

    return state_dict, release, checkpoint_name


def load_checkpoint(model, optimizer, lr_scheduler, load_arg='load', strict=True, load_only_weights=False):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)
    lora_load_dir = getattr(args, 'lora_load')

    if args.deepspeed:
        if not os.path.exists(load_dir):
            print_rank_0(f"WARNING: could not find the metadata file {load_dir}")
            print_rank_0(f" will not load any checkpoints and will start from random")
            return 0
        custom_load_fn, load_dir = get_custom_load_fn(model=model[0], load_dir=load_dir, lora_load_dir=lora_load_dir)
        if args.no_pipeline_parallel:
            load_zero_optim = sum(['zero' in file for file in os.listdir(load_dir)]) > 0
        else:
            load_zero_optim = sum(['global' in file for file in os.listdir(load_dir)]) > 0
        release = not load_zero_optim
        loaded_dir, state_dict = model[0].load_checkpoint(
            load_dir,
            # It is only loaded not strictly when lora is turned on and the original model is loaded.
            load_module_strict=not (release and is_enable_lora()),
            load_module_only=not load_zero_optim,
            load_optimizer_states=load_zero_optim,
            load_lr_scheduler_states=load_zero_optim,
            custom_load_fn=custom_load_fn
        )
        if loaded_dir is None:
            print_rank_0(f"WARNING: could not find the metadata file {load_dir}")
            print_rank_0(f" will not load any checkpoints and will start from random")
            return 0
        checkpoint_name = loaded_dir  # 开启lora时主要参数会从lora_load里读取，所以最后打印时用checkpoint_name传递
    else:
        unwrap_model_classes = (torchDDP, LocalDDP, Float16Module)
        if is_enable_lora():
            unwrap_model_classes += get_lora_model_classes()
        model = utils.unwrap_model(model, unwrap_model_classes)

        try:
            state_dict, release, checkpoint_name = get_state_dict_and_release(load_dir=load_dir,
                                                                              lora_load_dir=lora_load_dir)
        except ValueError as e:
            print_rank_0(f"{e}")
            return 0

    # set checkpoint version
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release or args.reset_iteration or load_only_weights:
        iteration = 0
        # Make DeepSpeed engine aware of this reset of iteration
        model[0].global_steps = 0
    else:
        iteration = load_iteration_from_state_dict(state_dict, checkpoint_name)

    # Check arguments.
    reset_train_valid_samples = args.reset_iteration
    if not load_only_weights and not reset_train_valid_samples:
        check_equal(args.consumed_train_samples, 0)
        check_equal(args.consumed_valid_samples, 0)
        if 'args' in state_dict:
            checkpoint_args = state_dict['args']
            check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args,
                                                  'consumed_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.consumed_valid_samples = getattr(checkpoint_args,
                                                  'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if not args.deepspeed:
        if is_enable_lora() and iteration == 0:
            strict = False
        if len(model) == 1:
            result = model[0].load_state_dict(state_dict['model'], strict=strict)
            if strict and result:
                print_rank_0(f"load checkpoint result:{result}")
        else:
            for i in range(len(model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not args.deepspeed:
        if not release and not args.finetune and not args.no_load_optim:
            load_optimizer_from_state_dict(optimizer, lr_scheduler, state_dict, checkpoint_name)

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            get_accelerator().set_rng_state(state_dict['cuda_rng_state'])
            # Check for empty states array
            if not state_dict['rng_tracker_states']:
                raise KeyError
            tensor_parallel.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {checkpoint_name} at iteration {iteration}')

    return iteration


def get_custom_load_fn(model, load_dir, lora_load_dir=None):
    custom_load_fn = None

    if isinstance(model, PipelineEngine):
        return custom_load_fn, load_dir

    if is_enable_lora():
        if lora_load_dir:
            custom_load_fn = get_lora_load_fn_with_deepspeed(model=model, base_model_load_dir=load_dir)
            load_dir = lora_load_dir
        else:
            custom_load_fn = lora_custom_load_fn_for_deepspeed
    return custom_load_fn, load_dir


def load_optimizer_from_state_dict(optimizer, lr_scheduler, state_dict, checkpoint_name):
    args = get_args()

    try:
        if optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])
        if lr_scheduler is not None and not args.no_load_lr_state:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    except KeyError:
        print_rank_0('Unable to load optimizer from checkpoint {}. '
                     'Specify --no-load-optim or --finetune to prevent '
                     'attempting to load the optimizer state, '
                     'exiting ...'.format(checkpoint_name))
        sys.exit()


def load_iteration_from_state_dict(state_dict, checkpoint_name):
    args = get_args()

    try:
        iteration = state_dict['iteration']
        if 'tokens' in state_dict:
            args.consumed_train_tokens = state_dict['tokens']
    except KeyError:
        try:  # Backward compatible with older checkpoints
            iteration = state_dict['total_iters']
        except KeyError:
            print_rank_0('A metadata file exists but unable to load '
                         'iteration from checkpoint {}, exiting'.format(
                checkpoint_name))
            sys.exit()
    return iteration


def load_state_dict_from_checkpoint_with_megatron(checkpoint_name, model_checkpoint_name=None):
    state_dict = torch.load(checkpoint_name, map_location='cpu')
    if model_checkpoint_name:
        model_state_dict = torch.load(model_checkpoint_name, map_location='cpu')
        state_dict = update_model_state_dict_with_megatron(model_state_dict=model_state_dict,
                                                           state_dict=state_dict)
    state_dict = handle_lora_modules_to_save_key_with_megatron(state_dict)
    return state_dict


def load_biencoder_checkpoint(model, only_query_model=False,
                              only_context_model=False, custom_load_path=None):
    """
    selectively load retrieval models for indexing/retrieving
    from saved checkpoints
    """

    args = get_args()

    model = utils.unwrap_model(model)

    load_path = custom_load_path if custom_load_path is not None else args.load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    checkpoint_name = get_checkpoint_name(load_path, iteration, False)
    if parallel_state.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ret_state_dict = state_dict['model']

    if only_query_model:
        ret_state_dict.pop('context_model')
    if only_context_model:
        ret_state_dict.pop('query_model')

    check_equal(len(model), 1)
    model[0].load_state_dict(ret_state_dict)
    torch.distributed.barrier()

    if parallel_state.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model
