import torch
import torch_npu
import ascendspeed_te_ops


def maybe_align(x, alignment_in_bytes=16):
    """Assume that x already has last dim divisible by alignment_in_bytes"""
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


class NPUDropoutAddRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x0, residual=None, rowscale=None, layerscale=None, return_dropout_mask=False):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        self.weight = maybe_align(self.weight.contiguous(), 16)
        self.bias = maybe_align(self.bias.contiguous(), 16) if self.bias is not None else None
        rowscale = maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
        layerscale = maybe_align(layerscale.contiguous(), 16) if layerscale is not None else None
        
        outputs = ascendspeed_te_ops.npu_dropout_add_layer_norm(
            x0, self.weight, residual, None, rowscale, layerscale,
            self.p, self.eps, self.prenorm, self.residual_in_fp32, True, return_dropout_mask)
        norm_result, pre_norm_result, mask_result = outputs
        return outputs
