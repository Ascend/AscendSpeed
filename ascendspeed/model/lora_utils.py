# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os

import torch


def get_lora_model_classes():
    from peft import PeftModel, LoraModel
    from .lora_modules import AscendLoraModel
    return PeftModel, LoraModel, AscendLoraModel


def is_enable_lora():
    from ascendspeed import get_args
    args = get_args()
    return bool(args.lora_target_modules)


def is_enable_lora_modules_to_save():
    from ascendspeed import get_args
    args = get_args()
    return is_enable_lora() and bool(args.lora_modules_to_save)


def is_module_name_in_lora_modules_to_save(module_name):
    from ascendspeed import get_args
    args = get_args()
    for modules_to_save_name in args.lora_modules_to_save:
        if module_name.endswith(f"{modules_to_save_name}.weight"):
            return True
    return False


def get_lora_state_dict(state_dict):
    from ascendspeed import get_args
    args = get_args()
    original_module_key = 'original_module.weight'
    modules_to_save_key = f'modules_to_save.{args.lora_adapter_name}.weight'

    @unwarp_dict
    def func_(key_name, state_dict_temp, state_dict):
        if "lora_" in key_name or key_name.endswith(original_module_key) or key_name.endswith(modules_to_save_key):
            state_dict_temp[key_name] = state_dict[key_name]

    return func_(state_dict)


def is_lora_state_dict(state_dict):
    @judge_unwarp_dict
    def func_(key):
        if "lora_" in key:
            return True
        return False

    return func_(state_dict)


def is_lora_modules_to_save_state_dict(state_dict):
    from ascendspeed import get_args
    args = get_args()
    modules_to_save_key = f'modules_to_save.{args.lora_adapter_name}'
    for key in state_dict.keys():
        if modules_to_save_key in key:
            return True
    return False


def handle_lora_modules_to_save_key(state_dict):
    if not is_enable_lora_modules_to_save():
        return state_dict
    if is_lora_modules_to_save_state_dict(state_dict):
        @unwarp_dict
        def func_(key_name, state_dict_temp, state_dict):
            if not is_module_name_in_lora_modules_to_save(key_name):
                state_dict_temp[key_name] = state_dict[key_name]

        return func_(state_dict)

    from ascendspeed import get_args
    args = get_args()
    original_module_key = 'original_module'
    modules_to_save_key = f'modules_to_save.{args.lora_adapter_name}'

    @unwarp_dict
    def func__(key_name, state_dict_temp, state_dict):
        state_dict_temp[key_name] = state_dict[key_name]
        if is_module_name_in_lora_modules_to_save(key_name):
            _key_name_list = key_name.split('.')
            if original_module_key not in key_name:
                original_module_name = '.'.join(_key_name_list[:-1] + [original_module_key] + [_key_name_list[-1]])
                state_dict_temp[original_module_name] = state_dict[key_name]
            if modules_to_save_key not in key_name:
                modules_to_save_name = '.'.join(_key_name_list[:-1] + [modules_to_save_key] + [_key_name_list[-1]])
                state_dict_temp[modules_to_save_name] = state_dict[key_name]

    return func__(state_dict)


def lora_custom_load_fn_for_deepspeed(src, dst):
    model = dst.get_base_model()
    state_dict = handle_lora_modules_to_save_key(state_dict=src)
    strict = is_lora_state_dict(state_dict=state_dict)
    # At this time, the model is a lora model, but the pre-training weights do not include lora, so strict is False
    result = model.load_state_dict(state_dict, strict=strict)
    if strict and result:
        from ascendspeed import print_rank_0
        print_rank_0(f"lora_custom_load_fn_for_deepspeed result:{result}")


def get_lora_load_fn_with_deepspeed(model, base_model_load_dir=None, tag=None):
    from deepspeed.runtime.state_dict_factory import SDLoaderFactory
    from deepspeed.runtime.pipe.module import PipelineModule

    if not base_model_load_dir:
        return lora_custom_load_fn_for_deepspeed

    if tag is None:
        latest_tag = "latest_universal" if model.load_universal_checkpoint() else "latest"
        latest_path = os.path.join(base_model_load_dir, latest_tag)
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()

    ckpt_list = model._get_all_ckpt_names(base_model_load_dir, tag)  # 需要在deepspeed外额外读取model的ckpt，故只能访问受保护成员
    sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine=model.checkpoint_engine)

    is_pipe_parallel = isinstance(model.module, PipelineModule)

    mp_rank = 0 if model.mpu is None else model.mpu.get_model_parallel_rank()
    load_path, checkpoint, _ = sd_loader.load(model.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel)

    if checkpoint is None:
        raise ValueError(f"failed to load {base_model_load_dir}.")

    module_state_dict = checkpoint['module']

    def _lora_load_fn(src, dst):
        state_dict = {}
        state_dict.update(module_state_dict)
        state_dict.update(src)
        return lora_custom_load_fn_for_deepspeed(src=state_dict, dst=dst)

    return _lora_load_fn


def get_lora_state_dict_with_deepspeed(model):
    original_state_dict = model.module.state_dict

    def _state_dict(destination=None, prefix='', keep_vars=False):
        state_dict = original_state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return get_lora_state_dict(state_dict=state_dict)

    return _state_dict


def handle_model_with_lora(model):
    from ascendspeed import get_args, print_rank_0
    from peft import LoraConfig, get_peft_model
    from . import lora_modules  # 给lora打补丁

    def _hook(_module, _x_in, _x_out):
        """ Extract the feature map of model"""
        _x_out.requires_grad_(True)

    def _create_hooks(_model, layer):
        """ Make the hooks function"""
        for name, module in _model.named_modules():
            _name = name.split('.')[-1]
            if _name in layer:
                module.register_forward_hook(_hook)

    args = get_args()

    model_len = len(model)
    for i in range(model_len):
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            modules_to_save=args.lora_modules_to_save,
            lora_dropout=0.0,
            bias="none",
        )
        model[i] = get_peft_model(model[i], lora_config)

        _create_hooks(model[i], args.lora_register_forward_hook)
        model[i].print_trainable_parameters()

    return model


def merge_dict(d1, d2):
    temp = {}
    for key in d1:
        temp[key] = d1[key]
    for key in d2:
        if key in d1:
            if isinstance(d2[key], dict):
                temp[key] = merge_dict(d1[key], d2[key])
            else:
                temp[key] = d2[key]
        else:
            temp[key] = d2[key]
    return temp


def update_model_state_dict_with_megatron(model_state_dict, state_dict):
    state_dict_ = {}
    for key in state_dict:
        if key.startswith('model'):
            state_dict_[key] = merge_dict(model_state_dict[key], state_dict[key])  # 更新model的权重

        else:
            state_dict_[key] = state_dict[key]
    return state_dict_


def unwarp_dict(func):
    def wrapper(state_dict):
        state_dict_temp = {}
        for key in state_dict:
            if isinstance(state_dict[key], dict):
                state_dict_temp[key] = wrapper(state_dict[key])
            else:
                # 对于tensor，可以改名，删除，增加
                func(key, state_dict_temp, state_dict)
        return state_dict_temp

    return wrapper


def judge_unwarp_dict(func):
    def wrapper(state_dict):
        for key in state_dict:
            if isinstance(state_dict[key], dict):
                r = wrapper(state_dict[key])
                if r:
                    return True
            else:
                r = func(key)
                if r:
                    return True
        return False

    return wrapper


def handle_lora_modules_to_save_key_with_megatron(state_dict):
    for key in state_dict:
        if key.startswith('model'):
            state_dict[key] = handle_lora_modules_to_save_key(state_dict[key])
    return state_dict
