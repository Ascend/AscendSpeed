"""Utility functions used throughout AscendSpeed core"""

import math
from functools import reduce
import operator

import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_attr_wrapped_model(model, attr, allow_none=True):
    """Get an attribute from a wrapped model"""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module
    return getattr(model, attr)


def get_model_type(model):
    return get_attr_wrapped_model(model, 'model_type')


def get_model_config(model):
    from ascendspeed import get_args
    from ascendspeed.arguments import core_transformer_config_from_args
    args = get_args()
    if args.deepspeed:
        return core_transformer_config_from_args(get_args())
    return get_attr_wrapped_model(model, 'config', allow_none=False)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                torch.empty(required_len,
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    with torch.no_grad():
        out.set_(inp.data)
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''
    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_
