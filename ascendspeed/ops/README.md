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
import ascendspeed

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
score = ascendspeed.npu_flash_attention(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, 
                scale, keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
```

single backward
```python
grad = ascendspeed.npu_flash_attention_grad(query, key, value, dy, head_num, input_layout, pse, padding_mask, 
                             atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, 
                             keep_prob, pre_tockens, next_tockens, gen_mask_parallel, sync)
```