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

import os
import re
import logging
import json
import pandas as pd
import tqdm

from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.llm_chat import LlmChat
from tasks.evaluation.eval_impl.template import GSM8K_TEMPLATE_DIR

logger = logging.getLogger(__name__)


class Gsm8kEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{fewshot_template}\n\n{question}",
                 output_template=r'The answer is (.*?) '):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = output_template

    def eval(self, llm_chat: LlmChat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        with open(GSM8K_TEMPLATE_DIR, encoding='utf-8') as f:
            gsm8k_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                gsm8k_list = json.load(f)
            subject_result = {}
            acc_n = 0
            for index, item in enumerate(gsm8k_list):
                instruction = self.instruction_template.format(fewshot_template=gsm8k_few_shot_template['few_shot'],
                                                               question=item['question'])
                chat_result, rank = llm_chat.chat(instruction=instruction, history=[])
                answer = None
                if chat_result:
                    answer = chat_result[0]
                    answer = answer.split('Q:')[0]
                    answer_result = answer.replace('$', '').replace(',', '') + '  '
                    answer_result = answer_result.replace('.', ' ', -1)
                try:
                    if rank == 0:
                        logger.info(instruction)
                        final_answer = re.findall(self.output_template, answer_result)
                        final_answer = [final_answer[0][::-1].replace('.', '', 1)[::-1]]
                        logger.info("correct: %s, answer : %s, reason: %s", item['answer'], final_answer, answer)
                        subject_result[str(index)] = final_answer
                        if subject_result[str(index)] == item['answer']:
                            acc_n += 1
                except Exception as e:
                    if rank == 0:
                        logger.info(e)
                    subject_result[str(index)] = str(e) + ". AI answer:" + answer
            if rank == 0:
                logger.info("Gsm8k dataset acc = %d/%d=%e", acc_n, len(gsm8k_list), acc_n / len(gsm8k_list))
                total_n += len(gsm8k_list)
                total_acc_n += acc_n
                answer_result['Gsm8k_dataset'] = subject_result
                score_datas.append(['Gsm8k_dataset', len(gsm8k_list), acc_n / len(gsm8k_list)])
        if rank == 0:
            logger.info("Gsm8k acc = %d/%d=%e", total_acc_n, total_n, total_acc_n / total_n)
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
