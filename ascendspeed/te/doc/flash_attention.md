# flash_attention对外接口
## 前向接口：
输入：
- query：必选输入，数据类型float16, bfloat16	
- key：必选输入，数据类型float16, bfloat16	
- value：必选输入，数据类型float16, bfloat16
（1）BNSD 下输入shape：query（b, n, s, d）   key（b, n, s, d） value（b, n, s, d） atten_mask (s, s,) alibi_mask（b, n, s,）
（2）BSH 下输入shape：query（b, s, h）   key（b, s, h） value（b, s, h） atten_mask (s, s,) alibi_mask（b, n, s,）
（3）SBH 下输入shape：query（s, b, h）   key（s, b, h） value（s, b, h） atten_mask (s, s,) alibi_mask（b, n, s,）
其中，d 必须是 <=128 并且为16的倍数； S 必须是 256 的倍数；	
- atten_mask：可选输入，数据类型float16，缺省none。
- alibi_mask：可选输入，数据类型float16, bfloat16，缺省none。

输出：
- attention_out：必选输出，数据类型float16, bfloat16

属性：
- scale_value：可选属性，数据类型float，缺省1。
- q_scale：可选属性，数据类型float，缺省1。
- head_num：可选属性，数据类型int64，缺省1。
- io_layout：可选属性，数据类型string	缺省“BNSD”。可支持“BSH”, “SBH”, “BNSD”
（1）BNSD 下输入shape：query（b, n, s, d）   key（b, n, s, d） value（b, n, s, d） atten_mask (s, s,) alibi_mask（b, n, s,）
（2）BSH 下输入shape：query（b, s, h）   key（b, s, h） value（b, s, h） atten_mask (s, s,) alibi_mask（b, n, s,）
（3）SBH 下输入shape：query（s, b, h）   key（s, b, h） value（s, b, h） atten_mask (s, s,) alibi_mask（b, n, s,）
其中，d 必须是 <=128 并且为16的倍数； S 必须是 256 的倍数；
- keep_prob：可选属性，数据类型float，默认值为1.0
- pre_tokens：可选属性，数据类型int64，默认值为2147483647
- next_tokens：可选属性，数据类型int64，默认值为1
- precise_mode：可选属性，数据类型int64，缺省0。0高精度，1高性能



## 反向接口
输入：
grad：必选输入，数据类型float16, bfloat16，正向attention_out的梯度输入

输出：
grad_query：必选输出，数据类型float16, bfloat16
grad_key：必选输出，数据类型float16, bfloat16	
grad_value：必选输出，数据类型float16, bfloat16	
（1）BNSD 下输入shape：grad_query（b, n, s, d）   grad_key（b, n, s, d） grad_value（b, n, s, d）
（2）BSH 下输入shape：grad_query（b, s, h）   grad_key（b, s, h） grad_value（b, s, h）
（3）SBH 下输入shape：grad_query（s, b, h）   grad_key（s, b, h） grad_value（s, b, h）
其中，d 必须是 <=128 并且为16的倍数； S 必须是 256 的倍数；


## 案例
```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import math
	import ascendspeed.te.ops
    
    batch_size, head_num, seq_size, head_dim = 1, 32, 2048, 128
    input_layout = "BNSD"
    keep_prob = 1.0
    pre_tokens = 65536
    next_tokens = 1
    scale_value = 0.088

    input_shape = (batch_size, head_num, seq_size, head_dim)
    softmax_shape = (batch_size, head_num, seq_size)
    mask_shape = (1, 1, seq_size, seq_size)
    dtype = torch.float16
    attention_score_grad = (torch.rand(input_shape).to(torch.float) - 0.5)
    query = (torch.rand(input_shape).to(torch.float) - 0.5) * 5
    key = (torch.rand(input_shape).to(torch.float) - 0.5) * 5
    value = (torch.rand(input_shape).to(torch.float) - 0.5) * 5

    atten_mask = np.tri(seq_size, k=-next_tokens) + np.tri(seq_size, k=-pre_tokens).transpose()
    alibi_mask = torch.zeros((batch_size, head_num, seq_size, seq_size)).to(torch.float)
    atten_mask = torch.from_numpy(atten_mask).to(torch.float)

    // 正向接口案例
	query.requires_grad = True
	key.requires_grad = True
	value.requires_grad = True
    result = ascendspeed.te.ops.ascend_flash_attention(query, key, value, atten_mask, None, None, scale_value, head_num, 0, keep_prob, pre_tokens, next_tokens, 0, -1)
    

    // 反向接口案例
	result.backward()
    query_grad = query.grad.data.cpu().to(torch.float).numpy()
    key_grad = key.grad.data.cpu().to(torch.float).numpy()
    value_grad = value.grad.data.cpu().to(torch.float).numpy()

```
