# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import time
import logging
import subprocess
 
import torch
from torch import distributed as dist
from deepspeed.accelerator import get_accelerator


logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)
 
 
def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task",
                       nargs='*',
                       default=[1, 2, 3], help='The task id to run.')
    group.add_argument("--local-rank", type=int, default=0)
    group.add_argument("--top-p", type=float, default=0.95, help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=50, help='Top k sampling.')
    group.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')
    group.add_argument("--max-length", type=int, default=256, help='Total length of text.')
    group.add_argument("--max-new-tokens", type=int, default=128, help='Size of the output generated text.')
    return parser


def task1(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """The most common usage"""
    if 1 not in list(map(int, args.task)):
        return

    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        instruction,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n===========================================")
        logging.info("\nYou:\n%s\n\nAscendSpeed:\n%s", prompt, output)
        logging.info("===========================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task2(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Returns the probability distribution of tokens"""
    if 2 not in list(map(int, args.task)):
        return

    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    tokens, log_probs = model.generate(
        instruction,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False,
        detokenize=False,
        return_output_log_probs=True
    )

    if dist.get_rank() == 0:
        logging.info("\n===========================================")
        logging.info("Probability distribution:\n%s", log_probs)
        logging.info("===========================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task3(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Interactive dialog mode with multiple rounds of conversation"""
    if 3 not in list(map(int, args.task)):
        return

    def get_context(content):
        res = system_template
        for q, r in content:
            if r is None:
                res += dialog_template.format(instruction=q)
            else:
                res += dialog_template.format(instruction=q) + r
        return res

    histories = []
    output = ""
    prompt = ""
    instruction = ""
    input_template = "You >> "
    response_template = "AscendSpeed:"
    command_clear = ["clear"]
    command_back = ["tput", "cup", "4", "0"]
    while True:
        terminate_runs = torch.zeros(1, dtype=torch.int64, device=torch.device(get_accelerator().device_name()))

        if dist.get_rank() == 0:
            if not histories:
                subprocess.call(command_clear)
                logging.info("===========================================================")
                logging.info("1. If you want to quit, please entry one of [q, quit, exit]")
                logging.info("2. To create new title, please entry one of [clear, new]")
                logging.info("===========================================================\n")

            prompt = input(input_template)
            if prompt.strip() in ["q", "exit", "quit"]:
                terminate_runs += 1

            if prompt.strip() in ["clear", "new"]:
                subprocess.call(command_clear)
                histories = []
                continue

            histories.append((prompt, None))
            instruction = get_context(histories)
            histories.pop()

        dist.all_reduce(terminate_runs)
        if terminate_runs > 0:
            break

        responses = model.generate(
            instruction,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )

        context = "\n"
        for q, r in histories:
            context += f"{input_template}{q}\n\n{response_template}\n{r}\n\n"

        for output in responses:
            if dist.get_rank() == 0:
                subprocess.call(command_back)
                logging.info("%s\n\n%s\n%s\n", context, response_template, output)

        histories.append((prompt, output))
