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

"""Transformer."""
from contextlib import nullcontext
from typing import Optional
import math
import logging

import numpy as np
import deepspeed
import torch
import torch_npu
import torch.nn.functional as F
from torch import einsum, nn
from torch import distributed as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.layer import MoE
from einops import rearrange

from ascendspeed import get_args
from ascendspeed.core.enums import ModelType
from ascendspeed import get_args, get_timers, get_num_microbatches, get_retro_args
from ascendspeed.core import utils, parallel_state, tensor_parallel
from ascendspeed.enums import PositionEmbeddingType
from ascendspeed.core.transformer.module.flash_attention import FlashSelfAttention
from ascendspeed.core.enums import AttnMaskType, LayerType, AttnType
from ascendspeed.model.fused_softmax import NPUFusedScaleMaskSoftmax
from ascendspeed.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb, apply_fused_rotary_pos_emb
from ascendspeed.core.transformer.module.triangle_attention import TriangleAttention
from ascendspeed.model.fused_bias_gelu import bias_gelu_impl
from ascendspeed.model.module import MegatronModule
from ascendspeed.model.utils import attention_mask_func, openai_gelu, erf_gelu, get_norm
from ascendspeed.error_utils import check_divisible, check_equal, ensure_valid

logger = logging.getLogger(__name__)

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
                        torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, moe=False,
                 enable_expert_tensor_parallelism=False):
        super(ParallelMLP, self).__init__()
        args = get_args()
        self.layer_fusion = args.mlp_layer_fusion
        self.add_bias = config.add_bias_linear
        ffn_hidden_size = config.ffn_hidden_size
        self.gated_linear_unit = config.gated_linear_unit
        if config.gated_linear_unit or self.layer_fusion:
            ffn_hidden_size *= 2
        ensure_valid(sum([args.mlp_layer_fusion, not args.no_add_gate, config.gated_linear_unit]) <= 1,
                     f"only can use one method in [mlp_layer_fusion :{args.mlp_layer_fusion},add_gate :"
                     f"{not args.no_add_gate},gated_linear_unit :{config.gated_linear_unit}],")
        self.add_gate = not args.no_add_gate and not self.layer_fusion and not self.gated_linear_unit
        if self.layer_fusion and not self.gated_linear_unit:
            self.proj = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )
        if self.add_gate:
            self.gate_proj = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )
        if not self.layer_fusion:
            # Project to 4h. If using swiglu double the output width
            self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=self.add_bias,
                gather_output=False,
                skip_bias_add=True,
                moe=moe,
                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
            )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]

            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)

            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
        )

    def forward(self, hidden_states):

        if self.add_gate:
            intermediate_parallel = F.silu(
                self.gate_proj(hidden_states)[0]) * self.dense_h_to_4h(hidden_states)[0]
        elif self.layer_fusion:
            gate_and_up_proj = self.proj(hidden_states)[0]
            (gate, up_proj) = tensor_parallel.utils.split_tensor_along_last_dim(
                gate_and_up_proj, 2, contiguous_split_chunks=True)
            intermediate_parallel = self.activation_func(gate) * up_proj
        else:
            # [s, b, 4hp]
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
            if self.bias_gelu_fusion:
                ensure_valid(self.add_bias is True)
                # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
                ensure_valid(hasattr(self, "__flops__") or self.activation_func == F.gelu)
                intermediate_parallel = \
                    torch_npu.fast_gelu(intermediate_parallel + bias_parallel)
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, config):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(config.hidden_size, args.num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts):
            self.experts.append(ParallelMLP(config))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2)  # [s b 1]

        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2))  # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2))  # [s*b 1]
        max_ind = max_ind.view(-1)  # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices, :]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices, :] = output
            output_bias_total[local_indices, :] = output_bias

        output_total = output_total * max_prob
        output_bias_total = output_bias_total * max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total


class CoreAttention(MegatronModule):

    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = utils.divide(projection_size,
                                                      world_size)
        self.hidden_size_per_attention_head = utils.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = utils.divide(
            config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.scale_mask_softmax = NPUFusedScaleMaskSoftmax(
            config,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

        self.use_flash_attn = config.use_flash_attn
        if self.use_flash_attn:
            self.core_flash_attn = FlashSelfAttention(causal=True, softmax_scale=(1.0 / self.norm_factor),
                                                      attention_dropout=config.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask, alibi=None):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        if not self.use_flash_attn:
            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2],
                                           output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3],
                                       output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        if alibi is None:
            matmul_result = None
        else:
            matmul_result = alibi[:, :output_size[3]]

        if self.use_flash_attn:
            if alibi is not None:
                # [1, sq] ==> [b * np, 1, sq]
                matmul_result = matmul_result.unsqueeze(0).repeat(output_size[0], output_size[1], 1, 1)
            q, k, v = [rearrange(x, 's b h d -> s b (h d)').contiguous() for x in (query_layer, key_layer, value_layer)]
            context_layer = self.core_flash_attn((q, k, v, self.num_attention_heads_per_partition), matmul_result,
                                                 attention_mask)
        else:
            # Raw attention scores. [b * np, sq, sk]
            if alibi is None:
                q_trans = query_layer.transpose(0, 1).contiguous()
                k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
                matmul_result = torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)
            else:
                q_trans = query_layer.transpose(0, 1).contiguous()
                k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
                matmul_result = self.beta * matmul_result + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            attention_probs = self.scale_mask_softmax(attention_scores,
                                                      attention_mask)
            if self.bf16:
                attention_probs = attention_probs.bfloat16()

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            else:
                attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1),
                           value_layer.size(2),
                           query_layer.size(0),
                           value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0),
                                           output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                                   output_size[2], -1)

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + \
                                      (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel

        self.group_query_attention = args.group_query_attention
        self.num_query_groups = args.num_query_groups

        query_projection_size = config.kv_channels * config.num_attention_heads
        if self.group_query_attention:
            kv_projection_size = args.kv_channels * args.num_query_groups
        else:
            kv_projection_size = args.kv_channels * args.num_attention_heads
        self.use_flash_attention = args.use_flash_attn
        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = utils.divide(
            query_projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = utils.divide(
            config.num_attention_heads, world_size)

        if self.group_query_attention:
            if args.num_query_groups % world_size != 0:
                raise NotImplementedError('Currently the num_query_groups should be '
                                          'a multiple of the tensor parallel size')
            self.num_query_groups_per_partition = utils.divide(
                args.num_query_groups, world_size)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
        else:
            ensure_valid(attention_type == AttnType.cross_attn)

            if self.group_query_attention:
                raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
            ensure_valid(query_projection_size == kv_projection_size)

            self.query = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False)

            self.key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False)
        self.position_embedding_type = args.position_embedding_type
        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        if args.use_fused_rotary_pos_emb:
            self.apply_rotary_pos_emb = apply_fused_rotary_pos_emb
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.use_triangle_attn = args.triangle_attn
        if self.use_triangle_attn:
            self.scale_mask_softmax = NPUFusedScaleMaskSoftmax(
                config,
                self.attn_mask_type,
                args.masked_softmax_fusion,
                attention_mask_func,
                (1 / self.norm_factor))
            self.block_size = args.triangle_block_size
            self.triangle_attn = TriangleAttention(block_size=self.block_size,
                                                   masked_softmax_func=self.scale_mask_softmax)

        else:
            self.core_attention = CoreAttention(self.layer_number, config,
                                                self.attn_mask_type)

        # 适配internlm模型
        skip_bias_add = getattr(config, "row_parallel_linear_skip_bias_add", True)
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask,
                                        rotary_pos_emb=None):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
            else rotary_pos_emb

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask,
            q_pos_emb, k_pos_emb)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, num_attention_heads):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None, alibi=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_length
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)

                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
                is_first_step = True
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================
        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                        (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                        * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer,
             key_layer,
             value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                            * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)

            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
            query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1,
                                           self.hidden_size_per_attention_head)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)
            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)


        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            ensure_valid(batch_end <= inference_key_memory.size(1))
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            ensure_valid(sequence_end <= inference_key_memory.size(0))
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
            batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
            batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                        :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                          :sequence_end, batch_start:batch_end, ...]

            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding
                    # (only the last token in the sequence)
                    q_pos_emb = q_pos_emb[sequence_end - 1: sequence_end]
                else:
                    # In the first forward pass of inference,
                    # we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire
                    # prefix + to-be-generated output so
                    # we slice to just the prefix.
                    q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1 and not self.use_flash_attention:
            key_layer = repeat_interleave(
                key_layer, self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim=2)
            value_layer = repeat_interleave(
                value_layer, self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if self.use_triangle_attn:
                query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
                key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
                value_layer = value_layer.permute(1, 2, 0, 3).contiguous()
                q_pos_emb = q_pos_emb.permute(1, 2, 0, 3).contiguous()
                k_pos_emb = k_pos_emb.permute(1, 2, 0, 3).contiguous()
            query_layer = self.apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = self.apply_rotary_pos_emb(key_layer, k_pos_emb)



            # can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.use_triangle_attn:
            context_layer = self.triangle_attn(query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask, alibi)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def repeat_interleave(inputs, repeats, dim):
    shape = inputs.shape
    new_shape = shape[:dim + 1] + (repeats,) + shape[dim + 1:]
    out_shape = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
    return inputs.unsqueeze(dim + 1).expand(new_shape).reshape(out_shape)


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0., num_experts=1):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        # Layernorm on the input data.

        # Normalize the input data.
        self.input_layernorm = get_norm(config)

        # Self attention.
        self.self_attention = ParallelAttention(
            config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = get_norm(config)

        # Cross attention.
        if self.layer_type in (LayerType.decoder,
                               LayerType.retro_decoder,
                               LayerType.retro_decoder_with_retriever,
                               LayerType.retro_encoder):
            self.inter_attention = ParallelAttention(
                config,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = get_norm(config)

        # MLP
        self.num_experts = num_experts
        if self.num_experts <= 1:
            self.mlp = ParallelMLP(config)
        else:
            enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
            self.mlp = MoE(args.hidden_size,
                           ParallelMLP(config,
                                       enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                           num_experts=self.num_experts,
                           ep_size=args.moe_expert_parallel_size,
                           k=args.topk,
                           use_residual=(args.mlp_type == 'residual'),
                           capacity_factor=args.moe_train_capacity_factor,
                           eval_capacity_factor=args.moe_eval_capacity_factor,
                           min_capacity=args.moe_min_capacity,
                           drop_tokens=args.moe_token_dropping, use_tutel=args.use_tutel,
                           enable_expert_tensor_parallelism=enable_expert_tensor_parallelism)
        self.bias_dropout_add_exec_handler = torch.enable_grad

        if args.retro_add_retriever:
            retro_args = get_retro_args()
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = retro_args.retro_gpt_chunk_length
            self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        if layer_type == LayerType.retro_decoder_with_retriever:
            self.retriever = ParallelTransformer(
                config=config,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever'
        else:
            self.retriever = None
        # Alibi
        if args.position_embedding_type == PositionEmbeddingType.alibi:
            self.alibi = self._build_alibi_tensor(args.seq_length, args.num_attention_heads,
                                                  args.micro_batch_size).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                self.alibi = self.alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                self.alibi = self.alibi.to(torch.bfloat16)
        else:
            self.alibi = None

    def default_decoder_cross_attention(self,
                                        encoder_output,
                                        enc_dec_attn_mask,
                                        layernorm_input,
                                        layernorm_output,
                                        bias_dropout_add_func):
        '''Cross attention for a standard encoder-decoder model.'''

        # Attention.
        attention_output, attention_bias = \
            self.inter_attention(layernorm_output,
                                 enc_dec_attn_mask,
                                 encoder_output=encoder_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)

        # Bias-dropout-add.
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        # Layer norm.
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return layernorm_input, layernorm_output

    def retro_encoder_cross_attention(self,
                                      retriever_output,
                                      layernorm_input,
                                      layernorm_output,
                                      bias_dropout_add_func):
        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape  # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = layernorm_output.reshape(self.retro_retrieved_length,
                                                   -1,
                                                   self.retro_num_neighbors,
                                                   d)
        chunked_outputs_before_layer_norm = \
            layernorm_input.reshape(self.retro_retrieved_length, -1,
                                    self.retro_num_neighbors, d)  # [r, bs*l, k, d]

        # Per-chunk attention.
        layernorm_inputs = []
        layernorm_outputs = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:, :, k].contiguous()
            attention_output, attention_bias = \
                self.inter_attention(
                    chunked_output,  # Q (neighbor embedding)
                    None,
                    encoder_output=retriever_output)  # K, V (hidden act)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = chunked_output
            else:
                residual = chunked_outputs_before_layer_norm[:, :, k]

            # Re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    None if attention_bias is None else attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
                layernorm_inputs.append(layernorm_input)

            # Layer norm.
            layernorm_output = \
                self.post_inter_attention_layernorm(layernorm_input)
            layernorm_outputs.append(layernorm_output)

        # Concatenate layer norms.
        # layernorm_input : [r, k * bs * l, d]
        layernorm_input = \
            torch.stack(layernorm_inputs, dim=1).reshape(ns, bs, d)
        layernorm_output = \
            torch.stack(layernorm_outputs, dim=1).reshape(ns, bs, d)

        return layernorm_input, layernorm_output

    def retro_decoder_cross_attention(self,
                                      retriever_input,
                                      retriever_output,
                                      retriever_attn_mask,
                                      layernorm_input,
                                      layernorm_output,
                                      inference_params,
                                      bias_dropout_add_func):
        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            nc  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape
        nc = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.layer_type == LayerType.retro_decoder_with_retriever:
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
            else:
                chunked_output = layernorm_output  # [nc * m, bs, d]
            chunked_output = chunked_output \
                .reshape(nc, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * nc, d) \
                .contiguous()

            # Get Encoder Output
            retriever_output = self.retriever(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                retriever_output=chunked_output,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params)  # [r, k * bs * nc , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * nc, d)  # [r * k, bs * nc, d]

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = layernorm_output[pad:]
        padded_chunks = torch.nn.functional.pad(
            attending_chunks,
            (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
            'constant', 0)
        padded_chunked_output = padded_chunks \
            .reshape(nc, self.retro_chunk_length, bs, d) \
            .permute(1, 2, 0, 3)
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * nc, d).contiguous()

        # Encoder output.
        attention_output, attention_bias = \
            self.inter_attention(padded_chunked_output,
                                 None,
                                 encoder_output=retriever_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                None if attention_bias is None else attention_bias.expand_as(attention_output),
                torch.zeros_like(attention_output),
                self.hidden_dropout)
            layernorm_input = layernorm_input \
                .reshape(self.retro_chunk_length, bs, nc, d) \
                .permute(2, 0, 1, 3)  # [nc, m, bs, d]
            layernorm_input = layernorm_input.reshape(self.retro_chunk_length * nc, bs, d)
            layernorm_input = torch.nn.functional.pad(
                layernorm_input,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns]  # [ns, b, d]
            layernorm_input = layernorm_input + residual

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return retriever_output, layernorm_input, layernorm_output

    def forward(self, hidden_states, attention_mask=None,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [s, b, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(layernorm_output,
                                attention_mask,
                                inference_params=inference_params,
                                rotary_pos_emb=rotary_pos_emb,
                                alibi=self.alibi)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            layernorm_input, layernorm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type == LayerType.retro_encoder:
            layernorm_input, layernorm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type in (LayerType.retro_decoder,
                                 LayerType.retro_decoder_with_retriever):
            retriever_output, layernorm_input, layernorm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    inference_params,
                    bias_dropout_add_func)
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        moe_loss = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=layernorm_output.device, dtype=layernorm_output.dtype)

        if self.num_experts == 1:
            mlp_output, mlp_bias = self.mlp(layernorm_output)
        else:
            mlp_output, moe_loss, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = utils.make_viewless_tensor(inp=output,
                                                requires_grad=output.requires_grad,
                                                keep_graph=True)

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(mlp_output,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            return output, retriever_output, moe_loss
        else:
            return output, moe_loss

    @staticmethod
    def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
        """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                   :n - closest_power_of_2]

        slopes = torch.Tensor(get_slopes(num_attention_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

        # Select the part of the tensor that corresponds to our tensor parallel index.
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_index = parallel_state.get_tensor_model_parallel_rank()
        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        return alibi[0]


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(args, model_type, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    is_encoder_and_decoder_model = (model_type == ModelType.encoder_and_decoder)
    if model_type == ModelType.retro_encoder:
        num_layers = args.retro_encoder_layers
    elif parallel_state.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            ensure_valid(args.pipeline_model_parallel_split_rank is not None)

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            ensure_valid(args.encoder_num_layers % num_ranks_in_encoder == 0,
                         error_message='encoder_num_layers (%d) must be divisible '
                                       'by number of ranks given to encoder (%d)' % (
                                           args.encoder_num_layers, num_ranks_in_encoder))
            ensure_valid(args.decoder_num_layers % num_ranks_in_decoder == 0,
                         error_message='decoder_num_layers (%d) must be divisible '
                                       'by number of ranks given to decoder (%d)' % (
                                           args.decoder_num_layers, num_ranks_in_decoder))
            if parallel_state.is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                       and parallel_state.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            ensure_valid(args.num_layers == args.encoder_num_layers)
            ensure_valid(args.num_layers % args.transformer_pipeline_model_parallel_size == 0,
                         error_message='num_layers must be divisible by transformer_pipeline_model_parallel_size')

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                   and parallel_state.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


def _get_layer_type(model_type, default_layer_type, retro_layer_numbers,
                    layer_number):
    args = get_args()
    if args.retro_add_retriever and layer_number in retro_layer_numbers:
        if model_type == ModelType.retro_decoder:
            return LayerType.retro_decoder_with_retriever \
                if layer_number == retro_layer_numbers[0] \
                else LayerType.retro_decoder
        elif model_type == ModelType.retro_encoder:
            return LayerType.retro_encoder
        else:
            raise Exception("Unsupported model type, '%s'." % model_type)
    else:
        return default_layer_type


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """

    def forward(self, inputs, **kwargs):
        ensure_valid(torch.is_tensor(inputs) or isinstance(inputs, tuple))
        if not hasattr(self, '_args'):
            self._args = get_args()
        rotary_pos_emb = self._args.rotary_pos_emb if self._args.use_rotary_position_embeddings else None
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            hidden_states, attention_mask = inputs, self._args.attn_mask
            # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(hidden_states, attention_mask, **kwargs, rotary_pos_emb=rotary_pos_emb)[0]
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            # HACK: currently MoE model does not support pipeline parallel, so
            # here we just ignore the moe_loss returned by forward()
            return super().forward(*inputs, **kwargs, rotary_pos_emb=rotary_pos_emb)[0], attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0, num_experts=[1]):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl
        self.retro_add_retriever = args.retro_add_retriever

        # Store activation checkpoiting flag.
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers
        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel

        self.sequence_parallel = config.sequence_parallel
        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        # Number of layers.
        error_info = 'num_layers must be divisible by pipeline_model_parallel_size'
        check_divisible(args.num_layers, parallel_state.get_pipeline_model_parallel_world_size(), error_info)
        self.num_layers = _get_num_layers(args, model_type,
                                          layer_type == LayerType.decoder)

        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, config.num_layers)]

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            retro_layer_start = 6 if config.num_layers <= 15 else 9
            self.retro_layer_numbers = \
                np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
        if model_type == ModelType.retro_encoder:
            self.retro_layer_numbers = [1]

        # Transformer layers.
        if args.retro_add_retriever:
            ensure_valid(self.recompute_granularity != 'full', error_message="Full recompute not supported for Retro.")
            ensure_valid(args.transformer_impl == 'local',
                         error_message="Transformer engine does not support Retro layers.")

        def build_layer(layer_number, n_e):
            if args.transformer_impl == 'local':
                current_layer_type = _get_layer_type(
                    model_type, layer_type, self.retro_layer_numbers,
                    layer_number)
                return ParallelTransformerLayer(
                    config,
                    layer_number,
                    layer_type=current_layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    num_experts=n_e)
            else:
                raise Exception("do not support {} transformer impl".format(args.transformer_impl))

        if args.virtual_pipeline_model_parallel_size is not None:
            error_info = 'num_layers_per_stage must be divisible by ' \
                         'virtual_pipeline_model_parallel_size'
            check_divisible(args.num_layers, args.virtual_pipeline_model_parallel_size)
            ensure_valid(args.model_type != ModelType.encoder_and_decoder)
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = (parallel_state.get_virtual_pipeline_model_parallel_rank() *
                      (config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                     (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers))
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    parallel_state.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            expression = len(num_experts) == 1 or len(num_experts) == args.num_layers // args.expert_interval
            ensure_valid(expression, error_message='num_experts must be either a single value or' \
                                                   ' a list of the same length as the number of MoE layers')

            # Create the list of MoE experts
            if len(num_experts) == 1:
                num_experts = num_experts * (args.num_layers // args.expert_interval)

            # Build the layers
            self.layers = []
            for i in range(self.num_layers):
                layer_num = i + 1 + offset
                if layer_num % args.expert_interval == 0:
                    n_e = num_experts[(layer_num - 1) // args.expert_interval]
                else:
                    n_e = 1
                self.layers.append(build_layer(layer_num, n_e))
            self.layers = torch.nn.ModuleList(self.layers)

            # Update dropout rate for Retro encoder.
            if model_type == ModelType.retro_encoder:
                for layer in self.layers:
                    layer.self_attention.core_attention.attention_dropout.p = \
                        args.retro_encoder_attention_dropout
                    layer.hidden_dropout = args.retro_encoder_hidden_dropout

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = get_norm(config)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              rotary_pos_emb, is_first_microbatch):
        args = get_args()

        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                moe_losses = []
                for index in range(start, end):
                    layer = self._get_layer(index)
                    output = layer(x_, *args, **kwargs)
                    if isinstance(output, tuple):
                        x_, moe_loss = output
                    else:
                        x_ = output
                        moe_loss = torch.tensor(0.0, device=x_.device, dtype=x_.dtype, requires_grad=True)
                    moe_losses.append(moe_loss)
                return (x_, *moe_losses)

            return custom_forward

        if args.deepspeed and args.deepspeed_activation_checkpointing:
            moe_losses = []
            # Make sure memory is freed.
            tensor_parallel.reset_checkpointed_activations_memory_buffer()
            layer = 0
            while layer < self.num_layers:
                hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                    custom(layer, layer + self.checkpoint_num_layers), False,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
                moe_losses.extend(local_moe_losses)
                layer += self.checkpoint_num_layers

            return hidden_states, moe_losses
        else:
            moe_losses = []
            te_forward_kwargs = {}
            if self.recompute_method == 'uniform':
                # Uniformly divide the total number of Transformer layers and
                # checkpoint the input activation of each divided chunk.
                # A method to further reduce memory usage reducing checkpoints.
                layer = 0
                while layer < self.num_layers:
                    hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                        custom(layer, layer + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask,
                        encoder_output, enc_dec_attn_mask,
                        None, None, None, None, rotary_pos_emb)
                    moe_losses.extend(local_moe_losses)
                    layer += self.recompute_num_layers
            elif self.recompute_method == 'block':
                # Checkpoint the input activation of only a set number of individual
                # Transformer layers and skip the rest.
                # A method fully use the device memory removing redundant re-computation.
                for layer in range(self.num_layers):
                    if layer < self.recompute_num_layers:
                        hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                            custom(layer, layer + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
                    else:
                        hidden_states, *local_moe_losses = custom(layer, layer + 1)(
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)

                    moe_losses.extend(local_moe_losses)
            elif self.recompute_method == "custom":
                if len(args.recomputation_layer_num) == \
                        parallel_state.get_pipeline_model_parallel_world_size():
                    self.recomputation_layer_num = args.recomputation_layer_num
                else:
                    raise ValueError(f"`recomputation_layer_num` length must equal to PP stage number.")
                for layer in range(self.num_layers):
                    if layer < self.recomputation_layer_num[
                        parallel_state.get_pipeline_model_parallel_rank()]:
                        hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                            custom(layer, layer + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
                    else:
                        hidden_states, *local_moe_losses = custom(layer, layer + 1)(hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)

            else:
                raise ValueError("Invalid activation recompute method.")
            return hidden_states, moe_losses

    def set_input_tensor(self, input_tensor):
        """
        Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        if isinstance(input_tensor, (list, tuple)):
            self.input_tensor = input_tensor[0]
        else:
            self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [b,s,h]
        # Checks.
        if inference_params:
            ensure_valid(self.recompute_granularity is None,
                         error_message='inference does not work with activation checkpointing')
        # # Reza's note: DeepSpeed inference does not support transposes
        if not self.ds_inference:
            if self.pre_process:
                # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
                # If the input flag for fp32 residual connection is set, convert for float.
                if self.fp32_residual_connection:
                    hidden_states = hidden_states.transpose(0, 1).contiguous().float()
                # Otherwise, leave it as is.
                else:
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
            else:
                # See set_input_tensor()
                hidden_states = self.input_tensor
            if encoder_output is not None:
                encoder_output = encoder_output.transpose(0, 1).contiguous()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )
        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        # Forward layers.
        with rng_context:
            with nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0  # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                moe_losses = []
                if self.checkpoint_activations or self.recompute_granularity == 'full':
                    hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                                           attention_mask,
                                                                           encoder_output,
                                                                           enc_dec_attn_mask,
                                                                           rotary_pos_emb,
                                                                           is_first_microbatch)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                    }
                    forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                    forward_kwargs['retriever_input'] = retriever_input
                    forward_kwargs['retriever_output'] = retriever_output
                    forward_kwargs['retriever_attn_mask'] = retriever_attn_mask

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)
                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                        # First Retro decoder layer returns both hidden_states
                        # and retriever_output. Make retriever_output available
                        # to subsequence Retro layers.
                        if isinstance(hidden_states, tuple):
                            ensure_valid((len(hidden_states) == 2 or len(hidden_states) == 3))
                            if len(hidden_states) == 2:
                                if not self.ds_inference:
                                    hidden_states, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)
                            else:
                                forward_kwargs["retriever_output"] = hidden_states[1]
                                if not self.ds_inference:
                                    hidden_states, _, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

                # Final layer norm.
            if self.post_process and self.post_layer_norm:

                if not self.ds_inference:
                    # Reverting data format change [s b h] --> [b s h].
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
                hidden_states = self.final_layernorm(hidden_states)

            return (hidden_states, *moe_losses)


class LMHeadPipe(MegatronModule):
    """
    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
        config:
    """

    def __init__(self, hidden_size, vocab_size, config):
        args = get_args()
        super(LMHeadPipe, self).__init__()
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=hidden_size,
                                                            output_size=vocab_size,
                                                            bias=False,
                                                            config=config,
                                                            init_method=config.init_method, )

    def forward(self, inputs, **kwargs):
        ensure_valid(torch.is_tensor(inputs) or isinstance(inputs, tuple))
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        logits, _ = self.lm_head(hidden_states)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask
