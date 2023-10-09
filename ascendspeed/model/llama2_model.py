# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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

import torch
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

from ascendspeed import get_args
from ascendspeed import mpu
from ascendspeed.core import tensor_parallel, parallel_state, utils
from ascendspeed.model.module import MegatronModule, MegatronModuleForCausalLM, float16_to_fp32, fp32_to_float16
from ascendspeed.core.enums import AttnMaskType, AttnType
from ascendspeed.model.utils import init_method_normal, scaled_init_method_normal, attention_mask_func
from ascendspeed.mpu.mappings import scatter_to_sequence_parallel_region
from ascendspeed.model.fused_softmax import NPUFusedScaleMaskSoftmax
from ascendspeed.model.language_model import Pooler


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
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).half()
        hidden_states = self.weight * hidden_states

        return hidden_states


class Llama2LMHead(MegatronModule):
    """Causal LM head for Llama

    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method,
                 parallel_output=True):
        super(Llama2LMHead, self).__init__()
        args = get_args()
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.parallel_output = parallel_output
        self.sequence_parallel = args.sequence_parallel
        self.lm_head = mpu.ColumnParallelLinear(input_size=self.hidden_size,
                                                output_size=vocab_size,
                                                bias=False,
                                                gather_output=not self.parallel_output,
                                                skip_bias_add=True,
                                                init_method=self.init_method,
                                                sequence_parallel_enabled=args.sequence_parallel)

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1).contiguous() if self.sequence_parallel else inputs
        logits, _ = self.lm_head(inputs)
        logits = logits.transpose(0, 1).contiguous() if self.sequence_parallel else logits
        return logits


class Llama2LMHeadPipe(Llama2LMHead):

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


class Llama2Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        init_method: weight initialization method
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 init_method):
        super(Llama2Embedding, self).__init__()
        args = get_args()

        self.hidden_size = hidden_size
        self.init_method = init_method

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(vocab_size, self.hidden_size,
                                                          init_method=self.init_method)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, input_ids):
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        if self.sequence_parallel:
            embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings = scatter_to_sequence_parallel_region(embeddings)
            embeddings = embeddings.transpose(0, 1).contiguous()

        return embeddings


class Llama2EmbeddingPipe(Llama2Embedding):

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


class Llama2ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to intermediate
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method, moe=False, enable_expert_tensor_parallelism=False):
        self.mlp = Llama2ParallelMLP
        super(self.mlp, self).__init__()
        args = get_args()
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method

        # Project to intermediate.
        self.gate_proj = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            bias=False,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
            sequence_parallel_enabled=args.sequence_parallel
        )

        self.up_proj = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            bias=False,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
            sequence_parallel_enabled=args.sequence_parallel
        )

        self.activation_func = F.silu

        # Project back to h.
        self.down_proj = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
            sequence_parallel_enabled=args.sequence_parallel)

    def forward(self, hidden_states):
        intermediate_parallel = self.activation_func(self.gate_proj(hidden_states)[0]) * self.up_proj(hidden_states)[0]

        output, _ = self.down_proj(intermediate_parallel)
        return output


class Llama2ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.causal):
        super(Llama2ParallelAttention, self).__init__()

        assert attention_type == AttnType.self_attn
        assert attn_mask_type == AttnMaskType.causal

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

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = utils.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias=False,
                gather_output=False,
                init_method=self.init_method,
                sequence_parallel_enabled=self.sequence_parallel)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        self.scale_mask_softmax = NPUFusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            (1 / self.norm_factor))

        ## Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(self.hidden_size_per_attention_head)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            sequence_parallel_enabled=self.sequence_parallel)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
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
             value_layer) = utils.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Rotary Position Embedding
        # ==================================
        # [sq, b, np, hn] --> [b, np, sq, hn] TODO optimize the permute of dimension back and forth
        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        cos, sin = self.rotary_emb(value_layer, seq_len=new_tensor_shape[0])
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, offset=0)


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

        # =========================
        # Context layer. [sq, b, hp]
        # =========================
        context_layer = torch.matmul(attention_probs, value_layer)

        bs, nh, sq, hd = context_layer.shape
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.view(sq, bs, nh*hd)

        output, _ = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output


class Llama2ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number,
                 self_attn_mask_type=AttnMaskType.causal):
        args = get_args()

        super(Llama2ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        assert self_attn_mask_type == AttnMaskType.causal

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
        self.attention = Llama2ParallelAttention(
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
        self.mlp = Llama2ParallelMLP(self.init_method, self.output_layer_init_method)

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
                                       get_key_value=get_key_value)

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


class Llama2ParallelTransformerLayerPipe(Llama2ParallelTransformerLayer):
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


class Llama2ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True, post_process=True):

        super(Llama2ParallelTransformer, self).__init__()
        args = get_args()
        assert self_attn_mask_type == AttnMaskType.causal

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

        # Number of layers.
        assert args.num_layers % parallel_state.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // parallel_state.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return Llama2ParallelTransformerLayer(
                self.init_method,
                self.output_layer_init_method,
                layer_number)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
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
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

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

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states, attention_mask)
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


class Llama2ModelPipe(PipelineModule, MegatronModule, MegatronModuleForCausalLM):
    """llama Language model."""

    def __init__(self, parallel_output=True):
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
        self.specs.append(LayerSpec(Llama2EmbeddingPipe, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
                                    init_method=self.init_method, ))

        if args.fp32_residual_connection:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous().float())
        else:
            self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(Llama2ParallelTransformerLayerPipe
                          , init_method=self.init_method
                          , output_layer_init_method=self.output_layer_init_method
                          , layer_number=layer_idx))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(RMSNorm, args.hidden_size, eps=args.layernorm_epsilon,
                      sequence_parallel=args.sequence_parallel))

        self.specs.append(
            LayerSpec(Llama2LMHeadPipe, hidden_size=args.hidden_size, vocab_size=args.padded_vocab_size,
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


class Llama2Model(MegatronModule, MegatronModuleForCausalLM):
    """llaMA2 Language model."""

    def __init__(self, config, pre_process, post_process, parallel_output=True, add_pooler=False, **kwargs):
        super(Llama2Model, self).__init__(config=config, share_word_embeddings=False)
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
            self.embedding = Llama2Embedding(hidden_size=args.hidden_size,
                                             init_method=self.init_method,
                                             vocab_size=self.padded_vocab_size)

        # Transformer.
        self.language_model = Llama2ParallelTransformer(
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

            self.lm_head = Llama2LMHead(hidden_size=args.hidden_size,
                                       vocab_size=self.padded_vocab_size,
                                       init_method=self.init_method,
                                       parallel_output=self.parallel_output)

    def set_input_tensor(self, input_tensor):
        """See ascendspeed.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, attention_mask, labels=None, layer_past=None, get_key_value=False):
        args = get_args()

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
                    assert hidden_states.dtype == torch.half
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states, labels)
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(hidden_states.float(), labels)
                return loss

        return hidden_states