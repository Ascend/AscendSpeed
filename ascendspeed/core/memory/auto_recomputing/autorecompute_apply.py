from ascendspeed.core import tensor_parallel


def hook_checkpoint_forward(forward_func):
    def custom_forward(*args, **kargs):
        def inside_forward(*args):
            return forward_func(*args, **kargs)

        return tensor_parallel.checkpoint(inside_forward, None, *args)

    return custom_forward


def register_recursive_apply(layer_name, model, ctx):
    idx = 0
    if 'recompute' in ctx and ctx['recompute']:
        model.forward = hook_checkpoint_forward(model.forward)
    else:
        for name, module in model.named_children():
            next_name = layer_name + "." + name if layer_name != "" else name
            register_recursive_apply(next_name, module, ctx['layers'][idx])
            idx += 1
