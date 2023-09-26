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

import math
from functools import partial

import torch
import torch.nn.functional as F

from ascendspeed import get_args
from ascendspeed import mpu
from ascendspeed.core import parallel_state, utils
from ascendspeed.model.module import MegatronModule
from ascendspeed.core.enums import AttnMaskType, AttnType
from ascendspeed.model.utils import attention_mask_func
from ascendspeed.model.fused_softmax import NPUFusedScaleMaskSoftmax
from ascendspeed.model.triangle_attention import TriangleAttention
from ascendspeed.model import llama_model


class InternLMParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(InternLMParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        self.sequence_parallel = args.sequence_parallel
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.num_attention_heads = args.num_attention_heads
        projection_size = args.kv_channels * args.num_attention_heads
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = utils.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = utils.divide(
            args.num_attention_heads, world_size)
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size, 3 * projection_size, bias=True, gather_output=False,
                init_method=self.init_method, sequence_parallel_enabled=self.sequence_parallel)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.scale_mask_softmax = NPUFusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            (1 / self.norm_factor))

        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)
        self.use_triangle_attn = args.triangle_attn
        if self.use_triangle_attn:
            self.triangle_attn = TriangleAttention(block_size=1024,
                                                   masked_softmax_func=self.scale_mask_softmax)
        self.dense = mpu.RowParallelLinear(
            projection_size, args.hidden_size, bias=True, input_is_parallel=True,
            init_method=self.output_layer_init_method, skip_bias_add=False,
            sequence_parallel_enabled=self.sequence_parallel)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

        if self.attention_type == AttnType.self_attn:
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            (query_layer,
             key_layer,
             value_layer) = utils.split_tensor_along_last_dim(mixed_x_layer, 3)
        
        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        cos, sin = self.rotary_emb(value_layer, seq_len=new_tensor_shape[0])
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)
            
        if self.use_triangle_attn and layer_past is None:
            context_layer = self.triangle_attn(query_layer, key_layer, value_layer, attention_mask)
            output, _ = self.dense(context_layer)
            return output
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(3, 2))
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        context_layer = torch.matmul(attention_probs, value_layer)

        bs, nh, sq, hd = context_layer.shape
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(sq, bs, nh * hd)

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output

RotaryEmbedding = llama_model.RotaryEmbedding
apply_rotary_pos_emb = llama_model.apply_rotary_pos_emb
llama_model.LlamaParallelAttention = InternLMParallelAttention
InternModel = llama_model.LlamaModel
InternModelPipe = llama_model.LlamaModelPipe
