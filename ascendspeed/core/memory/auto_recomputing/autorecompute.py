import time

import torch
import torch.nn
import torch_npu

from ascendspeed import print_rank_0
from ascendspeed.arguments import parse_args
from ascendspeed.core import parallel_state
from .autorecompute_apply import hook_checkpoint_forward as checkpoint_forward
from .autorecompute_apply import register_recursive_apply as apply_autorecompute
from .autorecompute_solver import solve_graph


# design of auto-recompute

# # Workflow
# step 1: profile the computation process to gain the computation of each ``module``, using all recompute
# step 2: get the graph and solve the recompute plan
# step 3: apply the recompute plan

# Detail workflow
# step1: intercept the model to hook forward, step, and register pytorch forward hooks to get profiling information

# information needed:
# 1) forward time of all module,
# 2) memory consumption of checkpointed memory and all memory;
# 3) static memory size (after the first step)
# 4) # of layers (can be gained by total mem / per layer mem)

# step2: do the average computation of all graph
# step3: solve
# step4: apply
# 1) hook each module, change the hook forward of the recomputed module to use mpu.checkpoint
# 2) remove the forward hook to remove profiling function


class AutoRecompute:
    auto_recomputing = None

    def __init__(self):
        # layer profiling info
        self.context = {
            'module': []
        }
        # save origin modules
        self.checkpointed_modules = []
        # save modules hook, remove it after apply policy
        self.modules_hooks = []
        # current profiling step
        self.profiling_step = 0
        # step for stop profiling, default is 10
        self.stop_profiling_step = 10
        # min step for stop profiling
        self.min_profiling_step = 5
        # step for solve graph by auto recompute, after step for stop profiling
        self.solve_graph_at_step = 11
        # unit for device memory size(MB)
        self.unit_mb = 1024 * 1024

    @staticmethod
    def get_memory_status():
        used_memory = torch.npu.memory_allocated()
        reserved_memory = torch.npu.memory_reserved()
        return used_memory, reserved_memory

    def _cal_tensor_size(self, tensor):
        try:
            return tensor.numel() * tensor.element_size() / self.unit_mb
        except ZeroDivisionError:
            return 0

    def pre_hook_func(self, state, sync: bool, *args, **kargs):
        if sync:
            torch.npu.synchronize()
        used_memory, _ = self.get_memory_status()
        torch.npu.reset_max_memory_allocated()
        state['memory'] = used_memory
        state['time'] = time.time()
        size = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                size += self._cal_tensor_size(arg)
            elif isinstance(arg, tuple) or isinstance(arg, list):
                for t in arg:
                    if isinstance(t, torch.Tensor):
                        size += self._cal_tensor_size(t)
        state['input'] = size

    def post_hook_func(self, state, sync: bool, *args, **kargs):
        if sync:
            torch.npu.synchronize()
        used_memory, _ = self.get_memory_status()
        max_mem = torch.npu.max_memory_allocated()
        state['peak_memory'] = max_mem - state['memory']
        state['memory'] = (used_memory - state['memory']) // self.unit_mb
        if 'pre_total_time' in state:
            state['forward_cnt'] += 1
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] += state['time']
            try:
                state['time'] = state['pre_total_time'] / state['forward_cnt']
            except ZeroDivisionError:
                state['time'] = 0
        else:
            state['forward_cnt'] = 0
            state['time'] = (time.time() - state['time']) * 1000
            state['pre_total_time'] = 0

    def forward_pre_hook(self, name, parent_ctx, ctx):
        if self.profiling_step < self.stop_profiling_step:
            ctx['name'] = name
            if 'layers' in parent_ctx:
                parent_ctx['layers'].append(ctx)

        def hook(module, *args, **kargs):
            if self.profiling_step < self.stop_profiling_step:
                if 'module' in self.context:
                    self.context['module'].append(ctx)
                self.pre_hook_func(ctx, True, *args, **kargs)

        return hook

    def forward_post_hook(self, ctx):
        def hook(module, *args, **kargs):
            if self.profiling_step < self.stop_profiling_step:
                self.post_hook_func(ctx, True, *args)
                if 'module' in self.context:
                    self.context['module'].pop()

        return hook

    def register_recursive_hook(self, prefix_name, model, ctx):
        for name, module in model.named_children():
            if str.isdigit(name) and name != "0":
                # transformer layer
                module.no_checkpoint_forward = module.forward
                module.forward = checkpoint_forward(module.forward)
                self.checkpointed_modules.append(module)

            if 'layers' not in ctx:
                ctx['layers'] = []
            current_ctx = {}

            next_name = prefix_name + "." + name if prefix_name != "" else name
            pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(name, ctx, current_ctx))
            post_hook = module.register_forward_hook(self.forward_post_hook(current_ctx))
            self.modules_hooks.append(pre_hook)
            self.modules_hooks.append(post_hook)
            self.register_recursive_hook(next_name, module, current_ctx)

    def step_hook(self, model):
        self.profiling_step += 1
        if self.profiling_step == self.solve_graph_at_step:
            print_rank_0("AUTO-RECOMPUTE: solving recompute policy")
            print_rank_0("==================== AUTO-RECOMPUTE Report ====================")
            all_args = parse_args()
            solve_graph(self.context, parallel_state.get_pipeline_model_parallel_world_size(),
                        all_args.auto_recompute_device_size)
            print_rank_0("==================== AUTO-RECOMPUTE Report End ====================")
            for m in self.checkpointed_modules:
                m.forward = m.no_checkpoint_forward
            self.checkpointed_modules.clear()
            print_rank_0("AUTO-RECOMPUTE: applying policy to the model")
            apply_autorecompute("module", model, self.context)
            print_rank_0("AUTO-RECOMPUTE: applying policy to the model fin")
            for hook_handle in self.modules_hooks:
                hook_handle.remove()
            self.modules_hooks.clear()

    def hook_step_func(self, step_func, models):
        def custom_step_func(*args, **kargs):
            result = step_func(*args, **kargs)
            if self.profiling_step < self.stop_profiling_step:
                used_memory, reserved_memory = self.get_memory_status()
                self.context['used_mem'] = used_memory // self.unit_mb
            if isinstance(models, list):
                for model in models:
                    self.step_hook(model)
            else:
                self.step_hook(models)
            return result

        return custom_step_func

    def set_profiling_step(self, step):
        self.stop_profiling_step = step
        self.solve_graph_at_step = step + 1

    def is_enabled_auto_recompute(self):
        all_args = parse_args()
        if all_args.auto_recompute_device_size <= 0:
            return False
        if all_args.checkpoint_activations:
            print_rank_0("[ERROR] failed to start auto selective recompute train, please remove param: "
                         "\"checkpoint-activations\".")
            return False
        if all_args.auto_recompute_profiling_step < 5:
            print_rank_0("[ERROR] failed to start auto selective recompute train, please set param >=5 or remove it: "
                         "\"auto-recompute-profiling-step\".")
            return False

        self.set_profiling_step(all_args.auto_recompute_profiling_step)
        print_rank_0(
            "success to stat auto recompute train: auto-recompute-device-size={}, auto-recompute-profiling-step={}".format(
                all_args.auto_recompute_device_size, all_args.auto_recompute_profiling_step))
        return True


def get_auto_recomputing():
    if AutoRecompute.auto_recomputing is None:
        AutoRecompute.auto_recomputing = AutoRecompute()
    return AutoRecompute.auto_recomputing


def autorecompute_profile(setup_model_and_optimizer_func):
    def get_model_hook_func(*args, **kargs):
        models, optimizer, lr_scheduler = setup_model_and_optimizer_func(*args, **kargs)
        recomputing = get_auto_recomputing()
        if not recomputing.is_enabled_auto_recompute():
            return models, optimizer, lr_scheduler
        optimizer.step = recomputing.hook_step_func(optimizer.step, models)
        if isinstance(models, list):
            for model in models:
                recomputing.register_recursive_hook("module", model, recomputing.context)
        else:
            recomputing.register_recursive_hook("module", models, recomputing.context)
        print_rank_0("AUTO-RECOMPUTE: successfully hooking module")
        return models, optimizer, lr_scheduler

    return get_model_hook_func
