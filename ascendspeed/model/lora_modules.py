# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
import peft
from peft import LoraModel as PeftLoraModel
from peft.tuners.lora import LoraLayer, Embedding, Conv2d, Linear
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import transpose, _get_submodules, ModulesToSaveWrapper

from ascendspeed.mpu import RowParallelLinear, ColumnParallelLinear
from ascendspeed.error_utils import check_equal

check_equal(peft.__version__, "0.4.0")

if is_bnb_available():
    import bitsandbytes as bnb
    from peft.tuners.lora import Linear8bitLt, Linear4bit


class LoraParalleLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, is_paralle_a: bool = False):
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        self.is_paralle_a = is_paralle_a

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            if self.is_paralle_a:
                lora_a = RowParallelLinear(input_size=self.in_features, output_size=r, bias=False,
                                           input_is_parallel=kwargs.get('input_is_parallel', True), skip_bias_add=True,
                                           dtype=torch.float32)  # lora需要强制升格到32位精度，否则会溢出
                lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=False, dtype=torch.float32)
            else:
                lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
                lora_b = ColumnParallelLinear(input_size=r, output_size=self.out_features, bias=False,
                                              gather_output=kwargs.get('gather_output', False), dtype=torch.float32)
            self.lora_A.update(nn.ModuleDict({adapter_name: lora_a}))
            self.lora_B.update(nn.ModuleDict({adapter_name: lora_b}))

            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)


class LoraParallelLinear(ColumnParallelLinear, RowParallelLinear, LoraParalleLayer):
    """
    当目标层parallel_linear为RowParallelLinear时:
                -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
                     -   -
                   | a_1 |
                   | .   |
          lora_A = | .   |        lora_B = [ ... ]
                   | .   |
                   | a_p |
                    -   -
    为了保持输入、输出的shape一致，我们需要将lora的矩阵A进行行切分，而此时的lora_B则应该是完整的线性层;
    同理，当目标层是ColumnParallelLinear时，我们对lora_B进行列切分，而lora_A依然是完整的线性层。
    """

    def __init__(
            self,
            adapter_name: str,
            parallel_linear: Union[ColumnParallelLinear, RowParallelLinear],
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        self.parallel_linear_class = type(parallel_linear)
        parallel_linear_kwargs = {}
        if isinstance(parallel_linear, RowParallelLinear):
            parallel_linear_kwargs['input_is_parallel'] = parallel_linear.input_is_parallel
        else:
            parallel_linear_kwargs['gather_output'] = parallel_linear.gather_output
        type(parallel_linear).__init__(self, input_size=parallel_linear.input_size,
                                       output_size=parallel_linear.output_size, bias=parallel_linear.bias,
                                       skip_bias_add=parallel_linear.skip_bias_add,
                                       sequence_parallel_enabled=parallel_linear.sequence_parallel_enabled,
                                       **parallel_linear_kwargs)
        LoraParalleLayer.__init__(self, in_features=parallel_linear.input_size,
                                  out_features=parallel_linear.output_size,
                                  is_paralle_a=isinstance(parallel_linear, RowParallelLinear))

        # weight会在_replace_module函数中进行拷贝
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **parallel_linear_kwargs)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = False

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            result, bias = self.parallel_linear_class.forward(self, x)
            return result, bias
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result, bias = self.parallel_linear_class.forward(self, x)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result, bias = self.parallel_linear_class.forward(self, x)

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            lora_a = self.lora_A[self.active_adapter]
            lora_b = self.lora_B[self.active_adapter]
            lora_dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]

            lora_result = lora_a(lora_dropout(x))
            if isinstance(lora_result, tuple):
                lora_result = lora_result[0]
            lora_result = lora_b(lora_result)
            if isinstance(lora_result, tuple):
                lora_result = lora_result[0]
            lora_result = lora_result * scaling

            result = result + lora_result
        else:
            result, bias = self.parallel_linear_class.forward(self, x)

        result = result.to(previous_dtype)

        return result, bias


class AscendLoraModel(PeftLoraModel):
    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }

        new_module = self._create_new_bit_linear_module(target, adapter_name, bias, kwargs)
        if new_module is None:
            if isinstance(target, torch.nn.Embedding):
                embedding_kwargs = kwargs.copy()
                embedding_kwargs.pop("fan_in_fan_out", None)
                in_features, out_features = target.num_embeddings, target.embedding_dim
                new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
            elif isinstance(target, torch.nn.Conv2d):
                out_channels, in_channels = target.weight.size()[:2]
                kernel_size = target.weight.size()[2:]
                stride = target.stride
                padding = target.padding
                new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
            elif isinstance(target, (ColumnParallelLinear, RowParallelLinear)):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                new_module = LoraParallelLinear(adapter_name=adapter_name, parallel_linear=target, **kwargs)
            else:
                # 在_create_new_linear_module里还没有匹配上，会直接抛异常
                new_module = self._create_new_linear_module(target, adapter_name, lora_config, bias, kwargs)

        return new_module

    def _create_new_bit_linear_module(self, target, adapter_name, bias, kwargs):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        new_module = None
        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        return new_module

    def _create_new_linear_module(self, target, adapter_name, lora_config, bias, kwargs):
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
            )
        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)
        return new_module

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                elif isinstance(target, (ColumnParallelLinear, RowParallelLinear)):
                    parallel_linear_kwargs = {}
                    if target.parallel_linear_class is RowParallelLinear:
                        parallel_linear_kwargs['input_is_parallel'] = target.input_is_parallel
                        parallel_linear_kwargs['sequence_parallel_enabled'] = target.sequence_parallel_enabled
                    else:
                        parallel_linear_kwargs['gather_output'] = target.gather_output
                        parallel_linear_kwargs['sequence_parallel_enabled'] = target.sequence_parallel_enabled
                    new_module = target.parallel_linear_class(input_size=target.input_size,
                                                              output_size=target.output_size, bias=target.bias,
                                                              skip_bias_add=target.skip_bias_add,
                                                              **parallel_linear_kwargs)
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[peft.PeftType.LORA] = AscendLoraModel
