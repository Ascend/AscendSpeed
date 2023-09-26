# coding=utf-8
# Copyright (c) 2020, Huawei CORPORATION.  All rights reserved.
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

import ascendspeed.model.llama_model as llama_model
from ascendspeed import get_args
from ascendspeed import mpu
from ascendspeed.core import parallel_state
from ascendspeed.core.enums import AttnMaskType
from ascendspeed.model.llama_model import LlamaModel
from ascendspeed.model.llama_model import LlamaParallelTransformerLayer
from ascendspeed.model.llama_model import RMSNorm
from ascendspeed.model.module import MegatronModule


class BaichuanParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 self_attn_mask_type=AttnMaskType.causal,
                 pre_process=True, post_process=True):

        super(BaichuanParallelTransformer, self).__init__()
        args = get_args()

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

        # Number of layers.
        self.num_layers = args.num_layers // parallel_state.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return LlamaParallelTransformerLayer(
                self.init_method,
                self.output_layer_init_method,
                layer_number)

        if args.virtual_pipeline_model_parallel_size is not None:
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

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask, layer_past=None, get_key_value=False):
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
        idx = 0
        while idx < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(idx, idx + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            idx += self.checkpoint_num_layers

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
                hidden_states = mpu.checkpoint(
                    custom(idx, idx + 1),
                    hidden_states, attention_mask)
            else:
                hidden_states = custom(idx, idx + 1)(hidden_states, attention_mask)
        return hidden_states


class BaichuanModel(LlamaModel):
    """Baichuan Language model."""

    def __init__(self, config, pre_process, post_process, parallel_output=True, add_pooler=False):
        super(BaichuanModel, self).__init__(config, pre_process, post_process, parallel_output, add_pooler)
        # Transformer.
        self.language_model = BaichuanParallelTransformer(
            self.init_method,
            self.output_layer_init_method,
            self_attn_mask_type=self.self_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )


BaichuanModelPipe = llama_model.LlamaModelPipe
