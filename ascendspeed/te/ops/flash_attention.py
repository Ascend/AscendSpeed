import torch
import torch_npu
import ascendspeed_te_ops


class AscendFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, atten_mask_opt=None, alibi_mask_opt=None, scale_value=1.0, q_scale=1.0,
                head_num=1, io_layout='BNSD', keep_prob=1.0, pre_tokens=2147483647, next_tokens=1, precise_mode=0):
        
        layout = 0
        if io_layout == 'BSH':
            layout = 1
        if io_layout == 'SBH':
            layout = 2

        drop_mask, seed, offset, numels = ascendspeed_te_ops.npu_gen_mask(query, keep_prob, head_num, io_layout, True, False)

        outputs = ascendspeed_te_ops.npu_fa_forward(query, key, value, atten_mask_opt, alibi_mask_opt, drop_mask, 
                    scale_value, head_num, layout, keep_prob, pre_tokens, next_tokens, precise_mode, -1)

        ctx.scale_value = scale_value
        ctx.q_scale = q_scale
        ctx.head_num = head_num
        ctx.sync = False
        ctx.gen_mask_parallel = True
        ctx.layout = layout
        ctx.groups = -1
        ctx.keep_prob = keep_prob
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens
        ctx.precise_mode = precise_mode
        ctx.seed = seed
        ctx.offset = offset
        ctx.numels = numels

        ctx.save_for_backward(query, key, value, atten_mask_opt, alibi_mask_opt, outputs[0], outputs[1])

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        query, key, value, atten_mask_opt, alibi_mask_opt, attention_score, softmax_log_max_sum = ctx.saved_tensors
        drop_mask = ascendspeed_te_ops.npu_gen_mask_by_seed(query, ctx.keep_prob, ctx.gen_mask_parallel, ctx.sync,
                                                            ctx.seed, ctx.offset, ctx.numels)
        outputs = ascendspeed_te_ops.npu_fa_backward(grad_outputs, softmax_log_max_sum, attention_score, query, key, value, 
                                             atten_mask_opt, alibi_mask_opt, drop_mask, ctx.scale_value, ctx.head_num, 
                                             ctx.layout, ctx.keep_prob, ctx.pre_tokens, ctx.next_tokens, 
                                             ctx.precise_mode, ctx.groups)
        return outputs[0], outputs[1], outputs[2], None, None, None, None, None, None, None, None, None, None

ascend_flash_attention = AscendFlashAttention.apply
