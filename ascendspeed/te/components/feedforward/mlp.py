from dataclasses import dataclass

import torch
import torch.nn as nn

from ascendspeed.te.components import Activation, build_activation
from .base import Feedforward, FeedforwardConfig


@dataclass
class MlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    bias: bool


class MLP(Feedforward):
    r"""multi-layer perceptron"""
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: Activation,
        hidden_layer_multiplier: int,
        bias: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim_model
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_mlp, bias=bias),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)
