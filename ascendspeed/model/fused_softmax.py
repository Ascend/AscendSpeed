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

import torch
import torch_npu
import torch.nn as nn
from ascendspeed.core.enums import AttnMaskType
from ascendspeed.error_utils import check_equal, ensure_valid


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
            inputs, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(
            inputs, mask, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class NPUFusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
            self,
            config,
            attn_mask_type,
            scaled_masked_softmax_fusion,
            mask_func,
            scale,
        ):
        super(NPUFusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = config.fp16
        self.input_in_bf16 = config.bf16
        expression = not (self.input_in_fp16 and self.input_in_bf16)
        ensure_valid(expression, error_message="both fp16 and bf16 flags cannot be active at the same time.")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.scale = scale

        ensure_valid(self.scale is None or self.attention_softmax_in_fp32, error_message="softmax should be in fp32 when scaled")

    def forward(self, input_, mask):
        # [b, np, sq, sk]
        check_equal(input_.dim(), 4)

        if self.is_kernel_available(mask, *input_.size()):
            return self.npu_forward_fused_softmax(input_, mask)
        else:
            return self.npu_forward_torch_softmax(input_, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        return (
                self.scaled_masked_softmax_fusion  # user want to fuse
                and self.input_in_float16  # input must be fp16
                and 32 < sk <= 4096  # sk must be 32 ~ 4096
                and sq % 16 == 0  # sq must be divisor of 16
                and sk % 16 == 0  # sk must be divisor of 16
        )

    def npu_forward_fused_softmax(self, input_, mask):
        self.scale = self.scale if self.scale is not None else 1.0
        return torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)

    def npu_forward_torch_softmax(self, input_, mask):
        if self.input_in_float16 and self.attention_softmax_in_fp32:
            input_ = input_.float()

        if self.scale is not None:
            input_ = input_ * self.scale
        mask_output = self.mask_func(input_, mask) if mask is not None else input_
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.attention_softmax_in_fp32:
            probs = probs.half() if self.input_in_fp16 else probs.bfloat16()
        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)