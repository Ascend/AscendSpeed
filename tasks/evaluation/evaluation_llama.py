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
from ascendspeed.model.gpt_model import GPTModel
from ascendspeed.initialize import initialize_megatron
from ascendspeed.arguments import core_transformer_config_from_args
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.boolq_eval import BoolqEval
from tasks.evaluation.eval_impl.gsm8k_eval import Gsm8kEval
from tasks.evaluation.eval_impl.mmlu_eval import MmluEval
from tasks.evaluation.eval_impl.ceval_exam import CEvalExam
from tasks.evaluation.eval_impl.bbh_eval import BBHEval
from tasks.evaluation.eval_impl.agi_eval import AGIEvalExam
from tasks.evaluation.eval_impl.human_eval import HumanEval


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())

    if get_args().row_col_parallel_linear_bias:
        # internlm模型配置
        config.column_parallel_linear_bias = True
        config.row_parallel_linear_bias = True
        config.row_parallel_linear_skip_bias_add = False
    
    """Build the model."""
    init_model = GPTModel(
        config,
        parallel_output=False,
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
    group.add_argument("--row-col-parallel-linear-bias", action="store_true", default=False, 
                       help='Configuration for the InternLM model.')
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


class LLMChat(Chat):
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
            answer, score_df = mmlu_eval.eval(chat=agent)
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
            answer, score_df = gsm8k_eval.eval(chat=agent)
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
            answer, score_df = boolq_eval.eval(chat=agent)
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
            answer, score_df = ceval_exam.eval(chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def human_eval(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'human_eval' in path:
            data_path = path
    try:
        if data_path:
            human_eval_exam = HumanEval(test_dir=data_path)
            answer, score_df = human_eval_exam.eval(chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def agi_eval(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'agieval' in path:
            data_path = path
    try:
        if data_path:
            agieval_exam = AGIEvalExam(test_dir=data_path)
            answer, score_df = agieval_exam.eval(chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


def bbh_eval(eval_args, agent):
    data_path = None
    for path in eval_args.task_data_path:
        if 'bbh' in path:
            data_path = path
    try:
        if data_path:
            bbh = BBHEval(test_dir=data_path)
            answer, score_df = bbh.eval(chat=agent)
            logger.info(score_df)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path)
    max_new_tokens = args.max_new_tokens
    template = "{instruction}"
    if 'mmlu' in args.task:
        a = time.time()
        mmlu(args, LLMChat(args))
        logger.info(f'MMLU Running Time:, {time.time() - a}')
    if 'gsm8k' in args.task:
        a = time.time()
        gsm8k(args, LLMChat(args))
        logger.info(f'GSM8k Running Time: {time.time() - a}')
    if 'boolq' in args.task:
        a = time.time()
        boolq(args, LLMChat(args))
        logger.info(f'Boolq Running Time: {time.time() - a}')
    if 'ceval' in args.task:
        a = time.time()
        ceval(args, LLMChat(args))
        logger.info(f'Ceval Running Time: {time.time() - a}')
    if 'bbh' in args.task:
        a = time.time()
        bbh_eval(args, LLMChat(args))
        logger.info(f'bbh Running Time: {time.time() - a}')
    if 'agieval' in args.task:
        a = time.time()
        agi_eval(args, LLMChat(args))
        logger.info(f'agi_eval Running Time: {time.time() - a}')
    if 'human_eval' in args.task:
        a = time.time()
        human_eval(args, LLMChat(args))
        logger.info(f'Human_eval Running Time: {time.time() - a}')

