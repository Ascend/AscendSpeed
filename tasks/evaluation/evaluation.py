# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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
"""Sample Generate LLAMA"""
import os
import sys
import time
import logging

from torch import distributed as dist
from transformers import LlamaTokenizer
from ascendspeed import get_args
from ascendspeed.model import LlamaModel
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from tasks.evaluation.eval_api.llm_chat import LlmChat
from tasks.evaluation.eval_impl.boolq_eval import BoolqEval
from tasks.evaluation.eval_impl.gsm8k_eval import Gsm8kEval
from tasks.evaluation.eval_impl.mmlu_eval import MmluEval
from tasks.evaluation.eval_impl.ceval_exam import CEvalExam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logger = logging.getLogger(__name__)


def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())
    """Build the model."""
    init_model = LlamaModel(
        config,
        parallel_output=False,
        add_pooler=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return init_model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task-data-path",
                       nargs='*',
                       default=[],
                       help='Path to the training dataset. Accepted format:'
                            '1) a single data path, 2) multiple datasets in the'
                            'form: dataset1-path dataset2-path ...')
    group.add_argument("--temperature", type=float, default=0.5,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top-p", type=float, default=0.9,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--max-new-tokens", type=int, default=128,
                       help='Size of the output generated text.')
    group.add_argument("--task", nargs='*', default=[], help='Choose one task from mmlu, boolq and gsm8k')
    return parser


def get_result(result):
    if result:
        final_result = [result[0]]
        if result[1][0][tokenizer.encode("Yes")[-1]] >= result[1][0][tokenizer.encode("No")[-1]]:
            final_result.append('T')
        else:
            final_result.append('F')
    else:
        final_result = None
    return final_result


class LlamaChat(LlmChat):
    def __init__(self, llm_args):
        self.args = llm_args

    def chat(self, instruction, history):
        instruction_temp = template.format(instruction=instruction)
        result = model.generate(
            instruction_temp,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            stream=False,
            return_output_log_probs=True
        )
        return get_result(result), dist.get_rank()


def mmlu(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'mmlu' in path:
            data_path = path
    try:
        if data_path:
            mmlu_eval = MmluEval(test_dir=data_path)
            answer, score_df = mmlu_eval.eval(llm_chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def gsm8k(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'gsm8k' in path:
            data_path = path
    try:
        if data_path:
            gsm8k_eval = Gsm8kEval(test_dir=data_path)
            answer, score_df = gsm8k_eval.eval(llm_chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def boolq(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'boolq' in path:
            data_path = path
    try:
        if data_path:
            boolq_eval = BoolqEval(test_dir=data_path)
            answer, score_df = boolq_eval.eval(llm_chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def ceval(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'ceval' in path:
            data_path = path
    try:
        if data_path:
            ceval_exam = CEvalExam(test_dir=data_path)
            answer, score_df = ceval_exam.eval(llm_chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    model = LlamaModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path)
    max_new_tokens = args.max_new_tokens
    template = "{instruction}"
    if 'mmlu' in args.task:
        a = time.time()
        mmlu(args, LlamaChat(args))
        logger.info('MMLU Running Time: ', time.time() - a)
    if 'gsm8k' in args.task:
        a = time.time()
        gsm8k(args, LlamaChat(args))
        logger.info('GSM8k Running Time: ', time.time() - a)
    if 'boolq' in args.task:
        a = time.time()
        boolq(args, LlamaChat(args))
        logger.info('Boolq Running Time: ', time.time() - a)
    if 'ceval' in args.task:
        a = time.time()
        ceval(args, LlamaChat(args))
        logger.info('Ceval Running Time: ', time.time() - a)

