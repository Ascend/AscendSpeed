import torch
import torch_npu
import ascendspeed_te_ops


class AscendFusionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse=None, padding_mask=None, 
                atten_mask=None, scale=1., keep_prob=1., pre_tokens=2147483647, next_tokens=2147483647, 
                inner_precise=0, prefix=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
        
        outputs = ascendspeed_te_ops.npu_fusion_attention(query, 
                key, value, head_num, input_layout, pse, padding_mask, 
                atten_mask, scale, keep_prob, pre_tokens, next_tokens, 
                inner_precise, prefix, sparse_mode, gen_mask_parallel, sync)
        attention_in, softmax_max, softmax_sum, softmax_in, seed, offset, numels = outputs
        ctx.save_for_backward(
            query, key, value, pse, padding_mask, atten_mask, attention_in, softmax_max,
            softmax_sum, softmax_in)
        ctx.scale = scale
        ctx.input_layout = input_layout
        ctx.head_num = head_num
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.inner_precies = inner_precise
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync
        ctx.seed = seed
        ctx.offset = offset
        ctx.numels = numels
        ctx.prefix = prefix
        ctx.sparse_mode = sparse_mode

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        query, key, value, pse, padding_mask, atten_mask, attention_in, softmax_max,\
            softmax_sum, softmax_in = ctx.saved_tensors
        results = ascendspeed_te_ops.npu_fusion_attention_grad(
            query, key, value, grad_outputs, ctx.head_num, ctx.input_layout, pse, padding_mask, atten_mask,
            softmax_max, softmax_sum, softmax_in, attention_in, ctx.scale, ctx.keep_prob, ctx.pre_tokens,
            ctx.next_tokens, ctx.inner_precise, ctx.seed, ctx.offset, ctx.numels, ctx.prefix, ctx.sparse_mode,
            ctx.gen_mask_parallel, ctx.sync)
        
        return results, None, None, None, None, None, None, None, None, None, None, None, None, None


ascend_fusion_attention = AscendFusionAttention.apply
