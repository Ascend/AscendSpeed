# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from ascendspeed import get_args
from ascendspeed.model import GPTModel
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from tasks.inference.infer_base import task1, task2, task3, add_text_generate_args


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    config = core_transformer_config_from_args(get_args())
    init_model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        return_moe_loss=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    system_template = ""
    dialog_template = "{instruction}"
    template = system_template + dialog_template

    task1(args, model, system_template=system_template, dialog_template=dialog_template)
    task2(args, model, system_template=system_template, dialog_template=dialog_template)
    task3(args, model, system_template=system_template, dialog_template=dialog_template)
