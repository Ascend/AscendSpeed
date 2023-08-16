import torch
import torch_npu

import ascendspeed_ops_internel

from torch.autograd import Function
from torch.nn import Module


class _FlashAttention(Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, 
                pre_tockens, next_tockens, gen_mask_parallel, sync):
        flash_func = ascendspeed_ops_internel.npu_flash_attention_binding_forward
        attention_score = flash_func(ctx, query, key, value, head_num, input_layout, pse, padding_mask, 
                          atten_mask, scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)

        return attention_score

    @staticmethod
    def backward(ctx, grad_outputs):
        back_func = ascendspeed_ops_internel.npu_flash_attention_binding_backward
        results = back_func(ctx, grad_outputs)
        return results

class FlashAttention(Module):
    def __init__(self):
        super(FlashAttention, self).__init__()
        self.atten = _FlashAttention.apply

    def forward(self, query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, 
                scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, # max of int32
                gen_mask_parallel=True, sync=False):
        return self.atten(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                          scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)

def npu_flash_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
                        scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, # max of int32
                        gen_mask_parallel=True, sync=False):
    return ascendspeed_ops_internel.npu_flash_attention_forward(query, key, value, head_num, input_layout, 
                                    pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, 
                                    next_tockens, gen_mask_parallel, sync)

def npu_flash_attention_grad(query, key, value, dy, head_num, input_layout, pse=None, padding_mask=None, 
                             atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None, 
                             attention_in=None, scale_value=1., keep_prob=1., pre_tockens=2147483647, # max of int32
                             next_tockens=2147483647, gen_mask_parallel=True, sync=False):
    return ascendspeed_ops_internel.npu_flash_attention_backward(query, key, value, dy, head_num, input_layout, pse, 
                                    padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, 
                                    scale_value, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
