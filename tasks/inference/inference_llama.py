# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from ascendspeed import get_args
from ascendspeed.model import GPTModel
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from tasks.inference.infer_base import task_greedy_search, task_do_sample, task_beam_search, add_text_generate_args
from tasks.inference.infer_base import task_beam_search_with_sampling, task_return_output_log_probs, task_chat


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

    task_greedy_search(args, model, system_template=system_template, dialog_template=dialog_template)
    task_do_sample(args, model, system_template=system_template, dialog_template=dialog_template)
    task_beam_search(args, model, system_template=system_template, dialog_template=dialog_template)
    task_beam_search_with_sampling(args, model, system_template=system_template, dialog_template=dialog_template)
    task_return_output_log_probs(args, model, system_template=system_template, dialog_template=dialog_template)
    task_chat(args, model, system_template=system_template, dialog_template=dialog_template)
