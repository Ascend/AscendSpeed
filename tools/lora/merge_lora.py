# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from peft import PeftModel
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import deepspeed

from ascendspeed import get_args
from ascendspeed.checkpointing import save_checkpoint
from ascendspeed.model import GPTModel, Float16Module
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.utils import unwrap_model


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())
    init_model = GPTModel(
        config,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


if __name__ == "__main__":
    initialize_megatron(args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    GPTModel.from_pretrained(  # 会加载到args.model中
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    unwrap_classes = (torchDDP, LocalDDP, Float16Module)
    if args.deepspeed:
        unwrap_classes += (deepspeed.DeepSpeedEngine,)
    unwrapped_model = unwrap_model(args.model, unwrap_classes)[0]

    if not isinstance(unwrapped_model, PeftModel):
        raise ValueError(f"Model is not PeftModel.")  # 不支持deepspeed的pipeline模式(PeftModel已被unwrap)

    unwrapped_model.merge_and_unload()  # 原地修改
    merged_model = args.model  # save_checkpoint时需要没有unwrap的模型，所以使用args.model
    del args.model  # 但args里不能有model，不然hook函数没法序列化
    args.lora_target_modules = []  # 关闭lora
    print(merged_model)

    save_checkpoint(iteration=1, model=merged_model, optimizer=None, lr_scheduler=None)
