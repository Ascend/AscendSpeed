import torch
import torch_npu
import ascendspeed_te_ops


class AscendFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, 
                scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=1,
                gen_mask_parallel=True, sync=False):
        
        ctx.scale = scale
        ctx.input_layout = input_layout
        ctx.head_num = head_num
        ctx.pre_tokens = pre_tockens
        ctx.next_tokens = next_tockens
        ctx.inner_precies = inner_precise
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync

        outputs = ascendspeed_te_ops.npu_flash_attention(
            query, key, value, head_num, input_layout, pse, padding_mask, 
            atten_mask, scale, keep_prob, pre_tockens, next_tockens, 
            inner_precise, gen_mask_parallel, sync)

        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = outputs
        ctx.saved_for_backward(
            query, key, value, pse, padding_mask, atten_mask, attention_score, softmax_max,
            softmax_sum, softmax_out, seed, offset, numels
            )

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        query, key, value, pse, padding_mask, atten_mask, attention_score, softmax_max,\
            softmax_sum, softmax_out, seed, offset, numels = ctx.saved_tensors
        results = ascendspeed_te_ops.npu_flasg_attention_grad(
            query, key, value, grad_outputs, ctx.head_num, ctx.input_layout, pse, padding_mask, atten_mask,
            softmax_max, softmax_sum, softmax_out, attention_score, ctx.scale, ctx.keep_prob, ctx.pre_tokens,
            ctx.next_tokens, ctx.inner_precise, seed, offset, numels, ctx.gen_mask_parallel, ctx.sync)
        return results


ascend_flash_attention = AscendFlashAttention.apply
