import torch
from torch.optim.optimizer import Optimizer
from ascendspeed.error_utils import check_divisible_by_zero


class CAdamW(Optimizer):
    def __init__(self, params, lr=5e-6, beta=0.965, weight_decay=3e-2, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, eps=eps)
        super(CAdamW, self).__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            beta = group['beta']
            group_name = group['params']
            lr, weight_decay, eps = group['lr'], group['weight_decay'], group['eps']
            for _, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    device = p.device
                    state["exp_avg"] = torch.zeros(list(p.shape), dtype=torch.float16).to(device)
                    state["qs_val_1"] = torch.ones(list(p.shape)[:-1] + [1], dtype=torch.float32).to(device)
                state['step'] += 1
                grad = p.grad
                if group_name == 'no_weight_decay_params':
                    lr = 0.5 * lr
                    weight_decay = 0
                p.mul_(1 - lr * weight_decay)
                exp_avg_q = state['exp_avg']
                qs_val_1 = state["qs_val_1"]
                exp_avg = exp_avg_q.float() * qs_val_1
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                qs_val_1.copy_((torch.max(torch.abs(exp_avg), dim=-1, keepdim=True)[0]) / 65503.0 + eps)
                exp_avg_q.copy_((exp_avg * qs_val_1.reciprocal()).clamp_(-65503.0, 65503.0).half())
                p.add_(torch.sign(exp_avg), alpha=-lr)
        return loss
