# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn
import torch_npu
from ascendspeed import get_args


class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        args = get_args()
        self.use_fused_rmsnorm = args.use_fused_rmsnorm
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_fused_rmsnorm:
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
