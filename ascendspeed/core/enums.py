# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import enum


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2  # Overrides `attention_mask` to be a lower triangular matrix
    prefix = 3
    # Forces one to pass an `attention_mask` that's 1 if we need to mask.
    # Tensor that can be broadcast to [micro_batch_size, n_head, seq_length, seq_length]
    custom = 4
