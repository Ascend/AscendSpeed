# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Iterator, List, Union

import torch
import torch_npu
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ascendspeed import get_args
from ascendspeed.core import parallel_state
from ascendspeed.core.enums import ModelType
from ascendspeed.core.pipeline_parallel import p2p_communication
from ascendspeed.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from ascendspeed.global_vars import get_num_microbatches
from ascendspeed.model.lora_utils import get_lora_model_classes, is_enable_lora
from ascendspeed.utils import unwrap_model
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.model import Float16Module
from ascendspeed.error_utils import check_equal, check_type, ensure_var_is_none, ensure_var_is_not_none

# Types
Shape = Union[List[int], torch.Size]


def clear_npu_overflow_flag():
    float_status = torch.zeros(8).npu()  # 8 bit for overflow
    result = torch_npu.npu_clear_float_status(float_status)


def get_npu_overflow_flag():
    float_status = torch.zeros(8).npu()  # 8 bit for overflow
    result = torch_npu.npu_get_float_status(float_status)
    if float_status.cpu()[0] != 0:
        return True
    else:
        return False


def set_npu_overflow_flag():
    torch.tensor([65504]).half().npu() + 100  # fp16 overflow flag


def get_forward_backward_func():
    """
    Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    """

    args = get_args()
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            if args.foldx_mode == "fifo":
                forward_backward_func = forward_backward_pipelining_with_foldx_fifo
            if args.foldx_mode == "aiao":
                forward_backward_func = forward_backward_pipelining_with_foldx_aiao
            check_equal(get_num_microbatches() % args.pipeline_model_parallel_size, 0,
                        error_info='{} not equal {}: '
                                   'number of microbatches is not divisible by pipeline-parallel ' \
                                   'size when using interleaved schedule')
        elif args.optimized_pipeline:
            forward_backward_func = optimized_forward_backward_pipelining
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def get_forward_func():
    args = get_args()
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        elif args.optimized_pipeline:
            forward_backward_func = optimized_forward_backward_pipelining
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''
    Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    check_type(out, torch.Tensor)
    ensure_var_is_none(out._base, error_message="counter-productive to free a view of another tensor.")
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype, )


def custom_backward(output, grad_output):
    '''
    Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''
    check_equal(output.numel(), 1,
                error_info="{} not equal {}:output should be pseudo-'freed' in schedule, to optimize memory")
    check_type(output, torch.Tensor)
    check_type(grad_output, (torch.Tensor, type(None)),
               error_message="grad_output == '%s'." % type(grad_output).__name__)

    # Handle scalar output
    if grad_output is None:
        check_equal(output.numel(), 1, error_info="{} not equal {}:implicit grad requires scalar output.")
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format, )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
):
    """
    Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor.
    """
    args = get_args()
    if config.timers is not None and args.foldx_mode is None:
        config.timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True
    unwrap_model_classes = (torchDDP, LocalDDP, Float16Module)
    if is_enable_lora():
        unwrap_model_classes += get_lora_model_classes()
    unwrapped_model = unwrap_model(model, unwrap_model_classes)
    set_input_tensor = get_attr_wrapped_model(unwrapped_model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            if not args.no_pipeline_parallel:
                output_tensor = loss / num_microbatches
            else:
                output_tensor = loss
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None and args.foldx_mode is None:
        config.timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None):
    """
    Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage).
    """

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()
    if args.deepspeed:
        ensure_var_is_not_none(model)

    if config.timers is not None and args.foldx_mode is None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]
    clear_npu_overflow_flag()
    # Backward pass.
    if args.deepspeed:
        model.backward(output_tensor[0])
    else:
        if output_tensor_grad[0] is None and config.grad_scale_func is not None:
            output_tensor[0] = config.grad_scale_func(output_tensor[0])

        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)
    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None and args.foldx_mode is None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
):
    """
    Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        check_equal(len(model), 1,
                    error_info="{} not equal {}:non-pipeline-parallel schedule does not support model chunking")
        model = model[0]
    if isinstance(data_iterator, list):
        check_equal(len(data_iterator), 1,
                    error_info="{} not equal {}:non-pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    no_sync_func = config.no_sync_func
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    args = get_args()
    if args.deepspeed:
        model.set_gradient_accumulation_boundary(False)

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    overflow_flag_all = False
    with no_sync_func():
        for _ in range(num_microbatches - 1):
            output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                         input_tensor, forward_data_store, config, collect_non_loss_data)
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
            overflow_flag = get_npu_overflow_flag()
            overflow_flag_all = overflow_flag or overflow_flag_all
    if args.deepspeed:
        model.set_gradient_accumulation_boundary(True)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                 input_tensor, forward_data_store, config, collect_non_loss_data)

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
    overflow_flag = get_npu_overflow_flag()
    overflow_flag_all = overflow_flag or overflow_flag_all

    if overflow_flag_all:
        set_npu_overflow_flag()

    return forward_data_store


def forward_backward_pipelining_with_interleaving(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
):
    """
    Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise.
    """
    check_type(model, list, error_message="interleaved pipeline parallelism expected model chunking")
    for chunk in model:
        check_type(chunk, torch.nn.Module, error_message="invalid model chunking")
    check_type(data_iterator, list,
               error_message="interleaved pipeline parallelism expected each model chunk to have a data iterator")

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func(model[0].parameters())
        config.param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch):
        """
        Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step()).
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(
                    param_sync_microbatch_id
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """
        Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step()).
        """
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                    grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func(model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor = forward_step_helper(k, checkpoint_activations_microbatch)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                    k == (num_warmup_microbatches - 1)
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                    k == (num_warmup_microbatches - 1)
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    forward_k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(forward_k, checkpoint_activations_microbatch)

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(forward_k, checkpoint_activations_microbatch)

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )
            )

    # Launch any remaining grad reductions
    enable_grad_sync()
    if config.grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            config.grad_sync_func(params)

    return forward_data_store


def forward_backward_pipelining_with_foldx_aiao(*,
                                                forward_step_func,
                                                data_iterator: Union[Iterator, List[Iterator]],
                                                model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                num_microbatches: int,
                                                seq_length: int,  # unused
                                                micro_batch_size: int,  # unused
                                                decoder_seq_length: int = None,  # unused
                                                forward_only: bool = False,
                                                collect_non_loss_data: bool = False, ):
    """Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    output_tensor_grads = [[] for _ in range(len(model))]
    config = get_model_config(model[0])
    model_type = get_model_type(model[0])
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks

    num_chunk_warmup_microbatches = get_num_microbatches()
    num_warmup_microbatches = num_microbatches
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (num_chunk_warmup_microbatches * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // num_chunk_warmup_microbatches
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_first_stage() and \
                len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
            input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func=forward_step_func,
                                     data_iterator=data_iterator[model_chunk_id],
                                     model=model[model_chunk_id],
                                     num_microbatches=get_num_microbatches(),
                                     input_tensor=input_tensor, forward_data_store=losses_reduced, config=config)
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None)

        return input_tensor_grad

    def init_recv_prev(k):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_forward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                return False
        return True

    def init_recv_next(k):
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_backward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                return False
        return True

    input_tensors_ops = []
    output_tensor_grads_ops = []

    def gather_input_tensor(k):
        if not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=True) == 0):
            input_tensor, op = input_tensors_ops.pop(0)
            op.wait()
            input_tensors[get_model_chunk_id(k, forward=True)].append(
                p2p_communication.recv_gather(input_tensor))

    def gather_output_tensor_grad(k):
        if not (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=False) == (num_model_chunks - 1)):
            output_tensor_grad, op = output_tensor_grads_ops.pop(0)
            op.wait()
            output_tensor_grads[get_model_chunk_id(
                k, forward=False)].append(p2p_communication.recv_gather(output_tensor_grad))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    if not parallel_state.is_pipeline_first_stage():
        input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
        input_tensors_ops.append((input_tensor, ops[0]))
    for k in range(num_warmup_microbatches):
        # Determine if tensor should be received from previous stage.
        recv_prev = False if k == (num_microbatches - 1) else init_recv_prev(k)

        gather_input_tensor(k)

        if recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))

        output_tensor = forward_step_helper(k)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
        p2p_communication.async_communicate(output_tensor, None, False, False)

    model_gradient_reduces = []
    if not parallel_state.is_pipeline_last_stage():
        _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
        output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    for k in range(num_microbatches_remaining, num_microbatches):
        recv_next = init_recv_next(k)
        if k == (num_microbatches - 1):
            recv_next = False

        gather_output_tensor_grad(k)

        if get_model_chunk_id(k, forward=False) < num_model_chunks - 1 and \
                get_model_chunk_id(k, forward=False) < get_model_chunk_id(k - 1, forward=False):
            handles = model[get_model_chunk_id(k, forward=False) + 1].allreduce_gradients(async_op=True)
            model_gradient_reduces.append(handles)
        if recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

        input_tensor_grad = backward_step_helper(k)
        p2p_communication.async_communicate(None, input_tensor_grad, False, False)
    handles = model[0].allreduce_gradients(async_op=True)
    model_gradient_reduces.append(handles)
    for handles in model_gradient_reduces:
        for handle in handles:
            handle.wait()

    return losses_reduced


def get_tensor_shapes(
        *,
        rank: int,
        model_type: ModelType,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int,
        config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                    decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
):
    """
    Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise.
    """

    if isinstance(model, list):
        check_equal(len(model), 1,
                    error_info="{} not equal {}:non-interleaved pipeline parallelism does not support model chunking")
        model = model[0]
    if isinstance(data_iterator, list):
        check_equal(len(data_iterator), 1,
                    error_info="{} not equal {}:non-pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
            parallel_state.get_pipeline_model_parallel_world_size()
            - parallel_state.get_pipeline_model_parallel_rank()
            - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    i % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )
        send_forward(output_tensor, send_tensor_shapes, config)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = ((i + num_warmup_microbatches) % max_outstanding_backprops
                                                 ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func(model.parameters())

    return forward_data_store


def _get_tensor_shapes():
    args = get_args()
    tensor_shapes = []
    mbs = args.manual_mbs
    for m in mbs:
        tensor_shapes.append((args.seq_length, m, args.hidden_size))

    return tensor_shapes


def optimized_forward_backward_pipelining(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False, ):
    """Run non-interleaved 1F1B schedule, with reduced pipeline bubble.
    Returns dictionary with losses if the last stage, empty dict otherwise.
    """
    if isinstance(model, list):
        check_equal(len(model), 1,
                    error_info="{} not equal {}:"
                               "optimized_forward_backward_pipelining schedule does not support model chunking")
        model = model[0]
    if isinstance(data_iterator, list):
        check_equal(len(data_iterator), 1,
                    error_info="{} not equal {}:"
                               "optimized_forward_backward_pipelining schedule does not support model chunking")
        data_iterator = data_iterator[0]
    config = get_model_config(model)
    model_type = get_model_type(model)
    tensor_shapes = _get_tensor_shapes()
    cnt_fwd, cnt_bwd = 0, 0

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    for _ in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward(config=config,
                                                      tensor_shape=tensor_shapes[cnt_fwd])
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, losses_reduced, config)
        p2p_communication.send_forward(output_tensor, config=config)
        cnt_fwd += 1
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(config=config,
                                                      tensor_shape=tensor_shapes[cnt_fwd])

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, losses_reduced, config)
        if forward_only:
            p2p_communication.send_forward(output_tensor, config=config)
        else:
            output_tensor_grad = \
                p2p_communication.send_forward_recv_backward(output_tensor,
                                                             tensor_shape=tensor_shapes[cnt_bwd], config=config)

        cnt_fwd += 1

        # Add input_tensor and output_tensor to end of list, then pop from the
        # start of the list for backward pass.
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        if forward_only:
            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(config=config,
                                                              tensor_shape=tensor_shapes[cnt_fwd])
        else:
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(input_tensor, output_tensor,
                              output_tensor_grad, model_type, config, model)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, config=config)
            else:
                input_tensor = \
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad, tensor_shape=tensor_shapes[cnt_fwd], config=config)
        cnt_bwd += 1

    # Run cooldown backward passes.
    if not forward_only:
        for _ in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(
                tensor_shape=tensor_shapes[cnt_bwd], config=config)

            input_tensor_grad = \
                backward_step(input_tensor, output_tensor,
                              output_tensor_grad, model_type, config, model)

            p2p_communication.send_backward(input_tensor_grad, config)

            cnt_bwd += 1

    return losses_reduced


def forward_backward_pipelining_with_foldx_fifo(
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False, ):
    """Returns dictionary with losses if the last stage, empty dict otherwise."""

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    output_tensor_grads = [[] for _ in range(len(model))]
    config = get_model_config(model[0])
    model_type = get_model_type(model[0])
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    # Run all forward passes and then all backward passes if number of
    # microbatches is just the number of pipeline stages.
    # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
    # all workers, followed by more microbatches after depending on
    # stage ID (more forward passes for earlier stages, later stages can
    # immediately start with 1F1B).
    if get_num_microbatches() == pipeline_parallel_size:
        num_warmup_microbatches = num_microbatches
        all_warmup_microbatches = True
    else:
        num_warmup_microbatches = \
            (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (
            num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_microbatches = min(num_warmup_microbatches,
                                        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func=forward_step_func,
                                     data_iterator=data_iterator[model_chunk_id],
                                     model=model[model_chunk_id],
                                     num_microbatches=get_num_microbatches(),
                                     input_tensor=input_tensor, forward_data_store=losses_reduced, config=config)
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model=None)

        return input_tensor_grad

    def init_recv_prev(k):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_forward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                return False
        return True

    def init_recv_next(k):
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            if k - (pipeline_parallel_size - 1) < 0:
                return False
            next_backward_model_chunk_id = get_model_chunk_id(
                k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                return False
        return True

    input_tensors_ops = []
    output_tensor_grads_ops = []

    def gather_input_tensor(k):
        if not (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=True) == 0):
            input_tensor, op = input_tensors_ops.pop(0)
            op.wait()
            input_tensors[get_model_chunk_id(k, forward=True)].append(
                p2p_communication.recv_gather(input_tensor))

    def gather_output_tensor_grad(k):
        if not (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and
                get_model_chunk_id(k, forward=False) == (num_model_chunks - 1)):
            output_tensor_grad, op = output_tensor_grads_ops.pop(0)
            op.wait()
            output_tensor_grads[get_model_chunk_id(k, forward=False)].append(
                p2p_communication.recv_gather(output_tensor_grad))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    if not parallel_state.is_pipeline_first_stage():
        input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
        input_tensors_ops.append((input_tensor, ops[0]))
    for k in range(num_warmup_microbatches):
        gather_input_tensor(k)
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        recv_prev = init_recv_prev(k)
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if output_tensor is not None:
            p2p_communication.async_communicate(output_tensor, None, False, False)
        if recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))
        if k == (num_warmup_microbatches - 1) and not all_warmup_microbatches:
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            if recv_next:
                _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
                output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        gather_input_tensor(forward_k)
        output_tensor = forward_step_helper(forward_k)
        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None
        # Determine if peers are sending, and where in data structure to put
        # received tensors.forward_only
        recv_prev = init_recv_prev(forward_k)
        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False
        if output_tensor is not None:
            p2p_communication.async_communicate(output_tensor, None, False, False)
        if recv_prev:
            input_tensor, _, ops = p2p_communication.async_communicate(None, None, True, False)
            input_tensors_ops.append((input_tensor, ops[0]))

        # Backward pass.
        backward_k = k
        gather_output_tensor_grad(backward_k)
        input_tensor_grad = backward_step_helper(backward_k)
        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if parallel_state.is_pipeline_first_stage():
            input_tensor_grad = None
        recv_next = init_recv_next(backward_k)
        if input_tensor_grad is not None:
            p2p_communication.async_communicate(None, input_tensor_grad, False, False)
        if recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))

    model_gradient_reduces = []
    # Run cooldown backward passes (flush out pipeline).
    if all_warmup_microbatches:
        if not parallel_state.is_pipeline_last_stage():
            _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    for k in range(num_microbatches_remaining, num_microbatches):
        gather_output_tensor_grad(k)
        if get_model_chunk_id(k, forward=False) < num_model_chunks - 1:
            if get_model_chunk_id(k, forward=False) < get_model_chunk_id(k - 1, forward=False):
                handles = model[get_model_chunk_id(k, forward=False) + 1].allreduce_gradients(async_op=True)
                model_gradient_reduces.append(handles)
        input_tensor_grad = backward_step_helper(k)
        recv_next = init_recv_next(k)
        if k == (num_microbatches - 1):
            recv_next = False
        if input_tensor_grad is not None:
            p2p_communication.async_communicate(None, input_tensor_grad, False, False)
        if recv_next:
            _, output_tensor_grad, ops = p2p_communication.async_communicate(None, None, False, True)
            output_tensor_grads_ops.append((output_tensor_grad, ops[0]))
    handles = model[0].allreduce_gradients(async_op=True)
    model_gradient_reduces.append(handles)
    for handles in model_gradient_reduces:
        for handle in handles:
            handle.wait()

    return losses_reduced
