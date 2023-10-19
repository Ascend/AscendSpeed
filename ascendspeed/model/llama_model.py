# coding=utf-8
# Copyright (c) 2023, Huawei Technologies Co., Ltd.  All rights reserved.
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
import logging
from functools import partial

import torch
try:
    import torch_npu
except ImportError:
    logging.warning("Import torch_npu Error.")

import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from ascendspeed import get_args
from ascendspeed.core import tensor_parallel, parallel_state, utils
from ascendspeed.enums import PositionEmbeddingType
from ascendspeed.model.module import MegatronModule, MegatronModuleForCausalLM, float16_to_fp32, fp32_to_float16
from ascendspeed.core.enums import AttnMaskType, LayerType, AttnType
from ascendspeed.model.utils import get_linear_layer, init_method_normal, scaled_init_method_normal, \
    attention_mask_func, \
    openai_gelu, erf_gelu
from ascendspeed.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from ascendspeed.model.fused_softmax import NPUFusedScaleMaskSoftmax
from ascendspeed.model.language_model import Pooler
from ascendspeed.model.triangle_attention import TriangleAttention
from ascendspeed.error_utils import check_equal, check_divisible


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (torch.tensor(base).double() ** (torch.arange(0, dim, 2).float().to(device) / dim).double())
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_fused_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


class RMSNorm(torch.nn.Module):  # for cpu
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, hidden_states):
        if self.weight.dtype == torch.float16:
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
            variance = torch.rsqrt(variance + self.variance_epsilon).half()
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            variance = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * variance
        hidden_states = self.weight * hidden_states
        return hidden_states


class LlamaLMHead(MegatronModule):
    """Causal LM head for Llama

    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
    """

    def __init__(self,
                 config,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True):
        super(LlamaLMHead, self).__init__()
        args = get_args()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output
        self.sequence_parallel = args.sequence_parallel
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                config=config,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=self.init_method,
                                                )

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1).contiguous() if self.sequence_parallel else inputs
        logits, _ = self.lm_head(inputs)
        logits = logits.transpose(0, 1).contiguous() if self.sequence_parallel else logits
        return logits


class LlamaLMHeadPipe(LlamaLMHead):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
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

        logits = super().forward(hidden_states)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask


class LlamaEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        init_method: weight initialization method
    """

    def __init__(self,
                 config,
                 hidden_size,
                 vocab_size,
                 init_method):
        super(LlamaEmbedding, self).__init__()
        args = get_args()

        self.hidden_size = hidden_size
        self.init_method = init_method

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(vocab_size, self.hidden_size,
                                                                      init_method=self.init_method, config=config)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        if self.sequence_parallel:
            embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings = scatter_to_sequence_parallel_region(embeddings)
            embeddings = embeddings.transpose(0, 1).contiguous()

        return embeddings


class LlamaEmbeddingPipe(LlamaEmbedding):

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[-1]

        embeddings = super().forward(input_ids)
        # If cmd args has attn_mask, we don't forward it as an activation.
        if not hasattr(self._args, 'attn_mask'):
            setattr(self._args, 'attn_mask', attention_mask)

        return embeddings


class LlamaParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to intermediate
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        super(LlamaParallelMLP, self).__init__()
        args = get_args()
        self.init_method = init_method
        self.layer_fusion = args.mlp_layer_fusion
        self.output_layer_init_method = output_layer_init_method
        self.col_parallel_linear = partial(
            tensor_parallel.ColumnParallelLinear,
            config=config,
            input_size=args.hidden_size,
            bias=False,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )
        # Project to intermediate.
        if self.layer_fusion:
            self.proj = self.col_parallel_linear(output_size=args.ffn_hidden_size*2)
        else:
            self.gate_proj = self.col_parallel_linear(output_size=args.ffn_hidden_size)
            self.up_proj = self.col_parallel_linear(output_size=args.ffn_hidden_size)

        self.activation_func = F.silu

        # Project back to h.
        self.down_proj = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            config=config,
            bias=False,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

    def forward(self, hidden_states):
        if self.layer_fusion:
            gate_and_up_proj = self.proj(hidden_states)[0]
            (gate, up_proj) = tensor_parallel.utils.split_tensor_along_last_dim(
                gate_and_up_proj, 2, contiguous_split_chunks=True)
            intermediate_parallel = self.activation_func(gate) * up_proj
        else:
            intermediate_parallel = self.activation_func(
                self.gate_proj(hidden_states)[0]) * self.up_proj(hidden_states)[0]

        output, _ = self.down_proj(intermediate_parallel)
        return output


class LlamaParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, config, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(LlamaParallelAttention, self).__init__()

        check_equal(attention_type, AttnType.self_attn)
        check_equal(attn_mask_type, AttnMaskType.causal)

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
        self.beta = 1.0
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number
        self.num_attention_heads = args.num_attention_heads
        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = utils.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = utils.divide(projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = utils.divide(args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            # 适配internlm
            bias = getattr(config, "column_parallel_linear_bias", False)
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                config=config,
                bias=bias,
                gather_output=False,
                init_method=self.init_method,
            )

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.scale_mask_softmax = NPUFusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            (1 / self.norm_factor))
        self.position_embedding_type = args.position_embedding_type
        if self.position_embedding_type != PositionEmbeddingType.alibi:
            # Rotary Position Embedding
            self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head,
                                              args.max_position_embeddings)
        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        if args.use_fused_rotary_pos_emb:
            self.apply_rotary_pos_emb = apply_fused_rotary_pos_emb

        self.use_triangle_attn = args.triangle_attn
        if self.use_triangle_attn:
            self.triangle_attn = TriangleAttention(block_size=1024,
                                                   masked_softmax_func=self.scale_mask_softmax)
        # 适配internlm模型
        bias = getattr(config, "row_parallel_linear_bias", False)
        skip_bias_add = getattr(config, "row_parallel_linear_skip_bias_add", True)
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            config=config,
            bias=bias,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=skip_bias_add,
        )

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def rotary_forward(self, hidden_states, attention_mask, layer_past, get_key_value):
        # rotary position, match 7b
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, 3 * h] --> 3 [sq, b, h]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.utils.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Rotary Position Embedding
        # ==================================
        # [sq, b, np, hn] --> [b, np, sq, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()
        cos, sin = self.rotary_emb(value_layer, seq_len=new_tensor_shape[0])
        query_layer, key_layer = self.apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # use triangle attention
        if self.use_triangle_attn and layer_past is None:
            context_layer = self.triangle_attn(query_layer, key_layer, value_layer, attention_mask)
            output, _ = self.dense(context_layer)
            return output

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        attention_scores = torch.matmul(query_layer, key_layer.transpose(3, 2))

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

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

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        if self.bf16:
            attention_probs = attention_probs.bfloat16()

        # =========================
        # Context layer. [sq, b, hp]
        # =========================
        context_layer = torch.matmul(attention_probs, value_layer)

        bs, nh, sq, hd = context_layer.shape
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(sq, bs, nh * hd)

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output

    def alibi_forward(self, hidden_states, attention_mask, layer_past, get_key_value, pse):
        # alibi position, match 13b
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, 3 * h] --> 3 [sq, b, h]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.utils.split_tensor_along_last_dim(mixed_x_layer, 3)
        # ==================================
        # Adjust key and value for inference
        # ==================================
        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)
        output_size = (query_layer.size(1), query_layer.size(2),
                       query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = pse[:output_size[0] * output_size[1], :, :output_size[3]]
        # Raw attention scores. [b * np, sq, sk]
        q_trans = query_layer.transpose(0, 1).contiguous()
        k_trans = key_layer.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * matmul_result + torch.bmm(q_trans, k_trans)
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

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

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)
        if self.bf16:
            attention_probs = attention_probs.bfloat16()
        # =========================
        # Context layer. [sq, b, hp]
        # =========================
        context_layer = torch.matmul(attention_probs, value_layer)

        bs, nh, sq, hd = context_layer.shape
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(sq, bs, nh * hd)

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]
        return output

    def forward(self, hidden_states, attention_mask, layer_past=None, get_key_value=False, pse=None):
        if self.position_embedding_type == PositionEmbeddingType.alibi:
            return self.alibi_forward(hidden_states, attention_mask, layer_past, get_key_value, pse)
        else:
            return self.rotary_forward(hidden_states, attention_mask, layer_past, get_key_value)


class LlamaParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method,
                 layer_number,
                 self_attn_mask_type=AttnMaskType.causal):
        args = get_args()

        super(LlamaParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        check_equal(self_attn_mask_type, AttnMaskType.causal)

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Layernorm on the input data.
        self.input_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.attention = LlamaParallelAttention(
            config,
            self.init_method,
            self.output_layer_init_method,
            layer_number,
            attn_mask_type=self_attn_mask_type)

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            sequence_parallel=args.sequence_parallel)

        # MLP
        self.rank = args.rank
        self.mlp = LlamaParallelMLP(config, self.init_method, self.output_layer_init_method)
        if args.position_embedding_type == PositionEmbeddingType.alibi:
            self.pse = self._build_alibi_tensor(args.seq_length, args.num_attention_heads,
                                                args.micro_batch_size).to(torch.cuda.current_device())
            if args.params_dtype == torch.float16:
                self.pse = self.pse.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                self.pse = self.pse.to(torch.bfloat16)
        else:
            self.pse = None

    @staticmethod
    def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
        # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/"
        # "a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
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

        alibi = alibi.repeat(batch_size, 1, 1)
        return alibi

    def forward(self, hidden_states, attention_mask=None,
                layer_past=None, get_key_value=False):
        # hidden_states: [b, s, h]
        residual = hidden_states
        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.input_layernorm(hidden_states)
        # Self attention.
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       layer_past=layer_past,
                                       get_key_value=get_key_value,
                                       pse=self.pse)

        if get_key_value:
            hidden_states, presents = hidden_states

        # Residual connection.
        hidden_states = hidden_states + residual
        residual = hidden_states

        # Layer norm post the self attention.
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP.
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        if get_key_value:
            hidden_states = [hidden_states, presents]
        return hidden_states


class LlamaParallelTransformerLayerPipe(LlamaParallelTransformerLayer):
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
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, '_args'):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError('Received more inputs than understood.')


class LlamaParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, config, init_method, output_layer_init_method,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True, post_process=True):

        super(LlamaParallelTransformer, self).__init__()
        args = get_args()
        check_equal(self_attn_mask_type, AttnMaskType.causal)

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.ds_inference = args.ds_inference
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers
        self.checkpoint_policy = args.checkpoint_policy
        self.checkpoint_block_layer = args.checkpoint_block_layer

        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel
        # Number of layers.
        error_info = 'num_layers must be divisible by pipeline_model_parallel_size'
        check_divisible(args.num_layers, parallel_state.get_pipeline_model_parallel_world_size(), error_info)
        self.num_layers = args.num_layers // parallel_state.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return LlamaParallelTransformerLayer(
                config,
                self.init_method,
                self.output_layer_init_method,
                layer_number)

        if args.virtual_pipeline_model_parallel_size is not None:
            error_info = 'num_layers_per_stage must be divisible by ' \
                         'virtual_pipeline_model_parallel_size'
            check_divisible(args.num_layers, args.virtual_pipeline_model_parallel_size, error_info)
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = parallel_state.get_virtual_pipeline_model_parallel_rank() * (
                    args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                     (parallel_state.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            offset = parallel_state.get_pipeline_model_parallel_rank() * self.num_layers

        self.layers = []
        # Build the layers
        for i in range(self.num_layers):
            layer_num = i + 1 + offset
            self.layers.append(build_layer(layer_num))

        self.layers = torch.nn.ModuleList(self.layers)

        if self.post_process:
            # Final layer norm before output.
            self.final_layernorm = RMSNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                sequence_parallel=args.sequence_parallel)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask=attention_mask)
                return x_

            return custom_forward

        # Make sure memory is freed.
        tensor_parallel.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = tensor_parallel.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                self.distribute_saved_activations,
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def _checkpointed_forward_block(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask=attention_mask)
                return x_

            return custom_forward

        # Make sure memory is freed.
        for idx in range(self.num_layers):
            if idx < self.checkpoint_block_layer:
                hidden_states = tensor_parallel.checkpoint(
                    custom(idx, idx + 1),
                    self.distribute_saved_activations,
                    hidden_states, attention_mask)
            else:
                hidden_states = custom(idx, idx + 1)(hidden_states, attention_mask)
        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        if isinstance(input_tensor, (list, tuple)):
            self.input_tensor = input_tensor[0]
        else:
            self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None, get_key_value=False):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # Reza's note: DeepSpeed inference does not support transposes
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

        if self.checkpoint_activations and self.checkpoint_policy == 'full':
            hidden_states = self._checkpointed_forward(hidden_states, attention_mask)
        elif self.checkpoint_activations and self.checkpoint_policy == 'block':
            hidden_states = self._checkpointed_forward_block(hidden_states, attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask=attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        if self.post_process:
            if not self.ds_inference:
                # Reverting data format change [s b h] --> [b s h].
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return output


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class LlamaModelPipe(PipelineModule, MegatronModule, MegatronModuleForCausalLM):
    """llama Language model."""

    def __init__(self, config, parallel_output=True):
        args = get_args()

        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.parallel_output = parallel_output

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        self.specs.append(LayerSpec(LlamaEmbeddingPipe, config=config, hidden_size=args.hidden_size,
                                    vocab_size=args.padded_vocab_size,
                                    init_method=self.init_method, ))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(LlamaParallelTransformerLayerPipe,
                          config=config,
                          init_method=self.init_method,
                          output_layer_init_method=self.output_layer_init_method,
                          layer_number=layer_idx))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(RMSNorm, args.hidden_size, eps=args.layernorm_epsilon,
                      sequence_parallel=args.sequence_parallel))

        self.specs.append(
            LayerSpec(LlamaLMHeadPipe, config=config, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
                      init_method=self.init_method, parallel_output=self.parallel_output)
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=parallel_state.get_pipeline_model_parallel_world_size(),
                                             num_mp=parallel_state.get_tensor_model_parallel_world_size(),
                                             num_dp=parallel_state.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method='type:transformer')


class LlamaModel(MegatronModule, MegatronModuleForCausalLM):
    """llama Language model."""

    def __init__(self, config, pre_process, post_process, parallel_output=True, add_pooler=False):
        super(LlamaModel, self).__init__(config, share_embeddings_and_output_weights=False)
        args = get_args()
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.hidden_size = args.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.add_pooler = add_pooler
        self.init_method = init_method_normal(args.init_method_std)
        self.output_layer_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        self.self_attn_mask_type = AttnMaskType.causal
        self.padded_vocab_size = args.padded_vocab_size

        if self.pre_process:
            self.embedding = LlamaEmbedding(config=config, hidden_size=args.hidden_size,
                                            init_method=self.init_method,
                                            vocab_size=self.padded_vocab_size)

        # Transformer.
        self.language_model = LlamaParallelTransformer(
            config,
            self.init_method,
            self.output_layer_init_method,
            self_attn_mask_type=self.self_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)

            self.lm_head = LlamaLMHead(config=config, hidden_size=args.hidden_size,
                                       vocab_size=self.padded_vocab_size,
                                       init_method=self.init_method,
                                       parallel_output=self.parallel_output)

    def set_input_tensor(self, input_tensor):
        """See ascendspeed.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, attention_mask, labels=None, layer_past=None, get_key_value=False, **kwargs):
        if self.pre_process:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = input_ids
        # decoder
        hidden_states = self.language_model(hidden_states, attention_mask, layer_past=layer_past,
                                            get_key_value=get_key_value)

        if self.post_process:
            if get_key_value:
                hidden_states, presents = hidden_states

            if self.add_pooler:
                hidden_states = self.pooler(hidden_states, pooling_sequence_index)

            hidden_states = self.lm_head(hidden_states)

            if labels is None:
                if get_key_value:
                    return [hidden_states, presents]
                return hidden_states

            else:
                if self.fp16_lm_cross_entropy:
                    check_equal(hidden_states.dtype, torch.half)
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states, labels)
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states.float(), labels)
                return loss

        return hidden_states

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """
        不能在这里调用self.state_dict函数，否则deepspeed保存时把state_dict挂成state_dict_for_save_checkpoint会循环调用
        """
        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
            prefix=prefix + 'language_model.', keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_.update(language_model_state_dict)
        # Save word_embeddings.
        if self.pre_process:
            embedding_state_dict = self.embedding.state_dict(prefix=prefix + 'embedding.', keep_vars=keep_vars)
            state_dict_.update(embedding_state_dict)
        if self.post_process:
            lm_head_state_dict = self.lm_head.state_dict(prefix=prefix + 'lm_head.', keep_vars=keep_vars)
            state_dict_.update(lm_head_state_dict)
        return state_dict_
