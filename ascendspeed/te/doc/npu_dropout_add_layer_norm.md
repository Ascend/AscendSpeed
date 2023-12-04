# npu_dropout_add_layer_norm 对外接口

CLASS NPUDropoutAddLayerNorm(hidden_size, prenorm=False, p=0.0, eps=1e-5, residual_in_fp32=False, dtype=None)

```
计算逻辑：
norm_result = LayerNorm(Dropout(x0 x rowscale x layerscale) + residual)
```

参数：
- hidden_size：必选属性，数据类型为int。表示输入x0最后一维（对这一维度做归一化）的大小。
- prenorm：可选属性，数据类型为bool，默认值：False。表示是否返回输出pre_norm_result。
- p：可选属性，数据类型float，默认值：0.。表示Dropout舍弃概率，eval模式下p=0.。
- eps：可选属性，数据类型float，默认值：1e-5。归一化处理时，添加到分母中的值，以提高数值稳定性。
- residual_in_fp32：可选属性，数据类型为bool，默认值：False。仅在输入residual为None时有意义。

***

CLASS NPUDropoutAddRMSNorm(hidden_size, prenorm=False, p=0.0, eps=1e-5, residual_in_fp32=False, dtype=None)

```
计算逻辑：
norm_result = RMSNorm(Dropout(x0 x rowscale x layerscale) + residual)
```

参数：
- hidden_size：必选属性，数据类型为int。表示输入x0最后一维（对这一维度做归一化）的大小。
- prenorm：可选属性，数据类型为bool，默认值：False。表示是否返回输出pre_norm_result。
- p：可选属性，数据类型float，默认值：0.。表示Dropout舍弃概率，eval模式下p=0.。
- eps：可选属性，数据类型float，默认值：1e-5。归一化处理时，添加到分母中的值，以提高数值稳定性。
- residual_in_fp32：可选属性，数据类型为bool，默认值：False。仅在输入residual为None时有意义。

***

ascendspeed_te_ops.npu_dropout_add_layer_norm(x0, weight, residual=None, bias=None, rowscale=None, layerscale=None, p=0., eps=1e-5, prenorm=False, residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False)

```
计算逻辑：
is_rms_norm=False：norm_result = LayerNorm(Dropout(x0 x rowscale x layerscale) + residual)
is_rms_norm=True：norm_result = RMSNorm(Dropout(x0 x rowscale x layerscale) + residual)
```

输入：
- x0：必选输入，shape：(B,S,H)。
- weight：必选输入，shape：(H,)。表示归一化处理时的权重参数。
- residual：可选输入，shape：(B,S,H)，默认值：None。表示残差。
- bias：可选输入，shape：(H,)，数据类型与输入weight一致，默认值：None。表示归一化处理时的偏置参数。
- rowscale：可选输入，shape：(B,S)，数据类型与输入x0一致，默认值：None。表示矩阵按行缩放比例。
- layerscale：可选输入，shape：(H,)，数据类型与输入x0一致，默认值：None。表示矩阵按列缩放比例。

```
支持的输入数据类型组合：
x0     residual   weight  norm_result
=====================================
fp32     fp32      fp32      fp32
fp16     fp32      fp32      fp16
fp16     fp16      fp32      fp16
bf16     fp32      fp32      bf16
bf16     bf16      fp32      bf16
fp16     fp16      fp16      fp16
bf16     bf16      bf16      bf16
```

输出：
- norm_result：必选输出，数据类型与输入x0一致。
- pre_norm_result：可选输出，数据类型与输入residual一致。
- mask_result：可选输出，数据类型为bool。

属性：
- p：可选属性，数据类型float，默认值：0.。表示Dropout舍弃概率，eval模式下p=0.。
- eps：可选属性，数据类型float，默认值：1e-5。归一化处理时，添加到分母中的值，以提高数值稳定性。
- prenorm：可选属性，数据类型为bool，默认值：False。表示是否返回输出pre_norm_result。
- residual_in_fp32：可选属性，数据类型为bool，默认值：False。仅在输入residual为None时有意义。
- is_rms_norm：可选属性，数据类型为bool，默认值：False。表示Normalization的类型，True - LayerNorm, False - RMSNorm。
- return_dropout_mask：可选属性，数据类型为bool，默认值：False。表示是否返回输出mask。

***
## 案例一：NPUDropoutAddLayerNorm 类的调用方式

- rowscale=None 且 layerscale=None
- 结果只返回 norm_result 
- is_rms_norm=False

```python
import torch
import torch_npu
from ascendspeed.te.ops.dropout_add_layer_norm import NPUDropoutAddLayerNorm

batch_size = 8
seqlen = 512
hidden_size = 192

# 正向接口案例
model = NPUDropoutAddLayerNorm(hidden_size, p=0.37, eps=1e-5,
                              residual_in_fp32=True, dtype=torch.float).npu()
torch.nn.init.normal_(model.weight)
torch.nn.init.normal_(model.bias)

x0 = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float, requires_grad=True)
residual = torch.randn_like(x0, dtype=torch.float, requires_grad=True)
norm_result, _, _ = model(x0.npu(), residual.npu())

# 反向接口案例
g = torch.randn_like(norm_result) / batch_size
norm_result.backward(g)
x0_grad = x0.grad
residual_grad = residual.grad
weight_grad = model.weight.grad
bias_grad = model.bias.grad
```

## 案例二：NPUDropoutAddRMSNorm 类的调用方式

- rowscale=None 且 layerscale=None
- 结果只返回 norm_result 
- is_rms_norm=True

```python
import torch
import torch_npu
from ascendspeed.te.ops.dropout_add_rms_norm import NPUDropoutAddRMSNorm

batch_size = 8
seqlen = 512
hidden_size = 192

# 正向接口案例
model = NPUDropoutAddRMSNorm(hidden_size, p=0.37, eps=1e-5,
                            residual_in_fp32=True, dtype=torch.float).npu()
torch.nn.init.normal_(model.weight)

x0 = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float, requires_grad=True)
residual = torch.randn_like(x0, dtype=torch.float, requires_grad=True)
norm_result, _, _ = model(x0.npu(), residual.npu())

# 反向接口案例
g = torch.randn_like(norm_result) / batch_size
norm_result.backward(g)
x0_grad = x0.grad
residual_grad = residual.grad
weight_grad = model.weight.grad
```

## 案例三：npu_dropout_add_layer_norm 函数的调用方式

- is_rms_norm=False

```python
import torch
import torch_npu
import ascendspeed_te_ops

batch_size = 8
seqlen = 512
hidden_size = 192

# 正向接口案例
x0 = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float, requires_grad=True)
residual = torch.randn_like(x0, dtype=torch.float, requires_grad=True)
weight = torch.ones(hidden_size, dtype=torch.float, requires_grad=True)
bias = torch.zeros(hidden_size, dtype=torch.float)
rowscale = torch.empty(batch_size, seqlen, dtype=torch.float)
survival_rate = 0.87
rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
layerscale = torch.randn(hidden_size, dtype=torch.float, requires_grad=True)

norm_result, pre_norm_result, mask_result = ascendspeed_te_ops.npu_dropout_add_layer_norm(
    x0.npu(),
    weight.npu(),
    residual.npu(),
    bias.npu(),
    rowscale.npu(),
    layerscale.npu(),
    0.37, # p
    1e-5, # eps
    True, # prenorm
    True, # residual_in_fp32
    False, # is_rms_norm
    True, # return_dropout_mask
)

# 反向接口案例
g = torch.randn_like(norm_result) / batch_size
(norm_result * torch.sigmoid(pre_norm_result)).backward(g)
x0_grad = x0.grad
residual_grad = residual.grad
weight_grad = weight.grad
layerscale_grad = layerscale.grad
```

## 案例四：NPUDropoutAddLayerNorm 类 + npu_dropout_add_layer_norm 函数的组合调用方式

- is_rms_norm=False

```python
import torch
import torch_npu
import ascendspeed_te_ops
from ascendspeed_te_ops import npu_dropout_add_layer_norm
from ascendspeed.te.ops.dropout_add_layer_norm import NPUDropoutAddLayerNorm

batch_size = 8
seqlen = 512
hidden_size = 192

# 正向接口案例
x0 = torch.randn(batch_size, seqlen, hidden_size, dtype=torch.float, requires_grad=True)
residual = torch.randn_like(x0, dtype=torch.float, requires_grad=True)
rowscale = torch.empty(batch_size, seqlen, dtype=torch.float)
survival_rate = 0.87
rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
layerscale = torch.randn(hidden_size, dtype=torch.float, requires_grad=True)

layer_norm_cls = NPUDropoutAddLayerNorm
layer_norm_func = npu_dropout_add_layer_norm
model = layer_norm_cls(hidden_size, prenorm=True, p=0.37, eps=1e-5,
                      residual_in_fp32=True, dtype=torch.float).npu()

torch.nn.init.normal_(model.weight)
torch.nn.init.normal_(model.bias)
norm_result, pre_norm_result, mask_result = layer_norm_func(
    x0.npu(),
    model.weight.npu(),
    residual.npu(),
    model.bias.npu(),
    rowscale.npu(),
    layerscale.npu(),
    model.p,
    model.eps,
    model.prenorm,
    model.residual_in_fp32,
    False, # is_rms_norm
    True, # return_dropout_mask
)

# 反向接口案例
g = torch.randn_like(norm_result) / batch_size
(norm_result * torch.sigmoid(pre_norm_result)).backward(g)
x0_grad = x0.grad
residual_grad = residual.grad
weight_grad = model.weight.grad
bias_grad = model.bias.grad
layerscale_grad = layerscale.grad
```