# coding=utf-8
# Copyright Information:
# Huawei Technologies Co., Ltd. All Rights Reserved © 2020
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


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=1., attention_dropout=0.):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkvn, pse, attention_mask):
        q, k, v, n = qkvn

        if self.causal:
            output = torch_npu.npu_fusion_attention(
                q, k, v, n, "SBH",
                pse=pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=self.softmax_scale,
                pre_tockens=k.shape[0],  # seq_len
                next_tockens=0,  # 0
                keep_prob=1 - self.dropout_p,
            )[0]
            return output
        raise Exception("the attention type {} is not support!".format(self.attention_type))