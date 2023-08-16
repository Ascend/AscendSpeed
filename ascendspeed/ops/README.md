# Install
```shell
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
python3 setup.py install
```

# Package
```shell
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
python3 setup.py build
python3 setup.py bdist_wheel
```

# Usage
After Installation, you can use it directly

```python
import torch
import torch_npu
import ascendspeed.ops

```

# Paramter
```python
# @par Inputs:  
# @li query: A matrix Tensor. The type support float16 and float32.
# @li key: A matrix Tensor. The type support float16 and float32.
# @li value: A matrix Tensor. The type support float16 and float32.
# @li dy: A matrix Tensor. The type support float16 and float32.  
# @li real_shift: A scalar. The type support float16 and float32.
# @li drop_mask: A matrix Tensor. The type support float16and float32.
# @li padding_mask: A matrix Tensor. The type support float16 and float32.
# @li atten_mask: A matrix Tensor. The type support float16 and float32.
# @li softmax_max: A matrix Tensor. The type support float16 and float32.  
# @li softmax_sum: A matrix Tensor. The type support float16 and float32.  
# @li softmax_in: A matrix Tensor. The type support float16 and float32.  
# @li attention_in: A matrix Tensor. The type support float16 and float32.
# @li scale_value: A mutable Tensor. Must met all of the following rules, shape of "keep_prob" should be (1,) or [1,].
# @li keep_prob: A bool. If True, changes the shape of "query" from [K, M] to [M, K].  
# @li query_transpose: A bool. If True, changes the shape of "key" from [N, K] to [K, N].
# @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to [K, N].  
# @li value_transpose: A bool. If True, changes the shape of "mid_data" from [K, M] to [M, K].  
# @li dy_transpose: A bool. If True, changes the shape of "value" from [N, K] to [K, N]. 
# @li is_transpose_attention: A bool. If True, changes the shape of "mid_data” from [K, M] to [M, K].
# @li is_flash: A bool. If True, changes the shape of "value" from [N, K] to [K, N].

# @par Outputs:
# dq: A matrix Tensor. The type support float16 and float32.
# dk: A matrix Tensor. The type support float16 and float32.  
# dv: A matrix Tensor. The type support float16 and float32.
# dpse: A matrix Tensor. The type support float16 and float32.
```

binding forward:
```python
flash = FlashAttention()
score = flash(self, query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
```

binding backward:
```python
query.requires_grad = True
score = flash(self, query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
score.backward()
query_grad = query.grad
```

single forward
```python
score = ascendspeed.ops.npu_flash_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
```

single backward
```python
grad = ascendspeed.ops.npu_flash_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, 
                             atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, 
                             keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
```
