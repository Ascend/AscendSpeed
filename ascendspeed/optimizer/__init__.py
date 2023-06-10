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

import math
import apex
import torch
from deepspeed.accelerator import get_accelerator

from ascendspeed import get_args
from ascendspeed.model import LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer


class AdamW(torch.optim.Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

apex.optimizers.FusedAdam = AdamW
apex.optimizers.FusedSGD = torch.optim.SGD

if get_accelerator().device_name() == 'cuda':
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD
else:
    from torch.optim import Adam
    from torch.optim import SGD

def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()

    weight_decay_params = {'params': [], 'name' : 'weight_decay_params'}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0, 'name': 'no_weight_decay_params'}
    
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                    if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params

def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
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
        else:
            raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    if args.deepspeed:
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
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
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad)
