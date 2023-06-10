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
from ascendspeed.model.enums import AttnMaskType


class NPUFusedScaleMaskSoftmax(torch.nn.Module):
    def __init__(
            self,
            input_in_fp16,
            input_in_bf16,
            attn_mask_type,
            scaled_masked_softmax_fusion,
            mask_func,
            softmax_in_fp32,
            scale,
    ):
        super(NPUFusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
                self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.mask_tri = None
        p = torch.npu.get_device_properties(0) if torch.npu.is_available() else None

        assert (
                self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input_, mask):
        # [b, np, sq, sk]
        assert input_.dim() == 4

        if torch.npu.is_available():
            return self.npu_forward_fused_softmax(input_, mask)

        return self.npu_forward_torch_softmax(input_, mask)

    def npu_forward_fused_softmax(self, input_, mask):
        if self.softmax_in_fp32:
            input_ = input_.float()

        if self.scale is None:
            self.scale = 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            if self.mask_tri is None:
                self.mask_tri = torch.triu(torch.ones(input_.shape, device=input_.device), diagonal=1).bool()
            probs = torch_npu.npu_scaled_masked_softmax(input_, self.mask_tri, self.scale, False)
        else:
            probs = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)

        probs = probs.half()

        return probs

    def npu_forward_torch_softmax(self, input_, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input_ = input_.float()

        if self.scale is not None:
            input_ = input_ * self.scale

        if self.attn_mask_type == AttnMaskType.causal:
            mask_tri = torch.triu(torch.ones(input_.shape, device=input_.device), diagonal=1).bool()
            mask_output = self.mask_func(input_, mask_tri)
        else:
            mask_output = self.mask_func(input_, mask) if mask is not None else input_
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            probs = probs.half() if self.input_in_fp16 else probs.bfloat16()
        return probs
