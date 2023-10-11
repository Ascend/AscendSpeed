# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from deepspeed.accelerator import get_accelerator
import torch
from torch.optim import SGD

from ascendspeed import get_args
from ascendspeed.model import LayerNorm
from ascendspeed.model.fused_layer_norm import MixedFusedLayerNorm
from ascendspeed.model.lora_utils import is_enable_lora
from ascendspeed.optimizer.adam import AdamW as Adam

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from .distrib_optimizer import DistributedOptimizer


def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()

    weight_decay_params = {'params': [], 'name': 'weight_decay_params'}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0, 'name': 'no_weight_decay_params'}
    
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm) or isinstance(module_, MixedFusedLayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                    if p is not None and p.requires_grad])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n != 'bias' and p.requires_grad])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias' and p.requires_grad])
    return weight_decay_params, no_weight_decay_params


def _get_sp_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay, sp-norm-without-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    params = 'params'
    name = 'name'
    args = get_args()

    weight_decay_params = {params: [], name: 'weight_decay_params'}
    no_weight_decay_params = {params: [], 'weight_decay': 0.0, name: 'no_weight_decay_params'}
    no_weight_decay_layernorm_params = {
        params: [],
        'weight_decay': 0.0,
        name: 'no_weight_decay_layernorm_sp_params'
    }

    def classify_params(local_module):
        nonlocal weight_decay_params
        nonlocal no_weight_decay_params
        nonlocal no_weight_decay_layernorm_params
        if isinstance(local_module, LayerNorm) or isinstance(local_module, MixedFusedLayerNorm):
            if getattr(list(local_module.named_parameters(recurse=False))[0][1], 'sequence_parallel', False):
                no_weight_decay_layernorm_params[params].extend(
                    [p for _, p in local_module.named_parameters(recurse=False)
                        if p is not None])
            else:
                no_weight_decay_params[params].extend(
                    [p for _, p in local_module.named_parameters(recurse=False)
                        if p is not None])
        else:
            for n, p in local_module.named_parameters(recurse=False):
                if p is not None and p.requires_grad:
                    if getattr(p, 'sequence_parallel', False):
                        no_weight_decay_layernorm_params[params].append(p)
                    elif 'bias' not in n:
                        weight_decay_params[params].append(p)
                    elif 'bias' in n:
                        no_weight_decay_params[params].append(p)

    for module in modules:
        for module_ in module.modules():
            classify_params(module_)

    return weight_decay_params, no_weight_decay_params, no_weight_decay_layernorm_params


def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    if args.deepspeed and args.sequence_parallel:
        param_groups = _get_sp_params_for_weight_decay_optimization(model)
    else:
        param_groups = _get_params_for_weight_decay_optimization(model)
    if args.create_moe_param_group:
        from deepspeed.moe.utils import is_moe_param, split_params_into_different_moe_groups_for_optimizer
        param_groups = split_params_into_different_moe_groups_for_optimizer(param_groups)
    
    if args.cpu_optimizer:
        assert args.optimizer == 'adam', 'CPU offloading is for Adam'
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay)
    else:
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps)
        elif args.optimizer == 'sgd':
            optimizer = SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.sgd_momentum)
        elif args.optimizer == 'fused_adam':
            from deepspeed.ops.adam.fused_adam import FusedAdam
            optimizer = FusedAdam(param_groups,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=(args.adam_beta1, args.adam_beta2),
                                 eps=args.adam_eps)

        else:
            raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    if args.deepspeed:
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local' and not is_enable_lora():
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        opt_ty = DistributedOptimizer if args.use_distributed_optimizer else Float16OptimizerWithFloat16Params
        return opt_ty(optimizer,
                      args.clip_grad,
                      args.log_num_zeros_in_grad,
                      params_have_main_grad,
                      args.use_contiguous_buffers_in_local_ddp,
                      args.fp16,
                      args.bf16,
                      args.params_dtype,
                      grad_scaler,
                      model)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad)
