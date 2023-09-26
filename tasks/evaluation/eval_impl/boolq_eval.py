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
import logging
import json
import pandas as pd
import tqdm
from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.llm_chat import LlmChat
logger = logging.getLogger(__name__)


class BoolqEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{passage}\nQuestion: {question}?\nAnswer:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template

    def eval(self, llm_chat: LlmChat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                boolq_question_list = []
                for line in f.readlines():
                    boolq_question_list.append(json.loads(line))
            subject_result = {}
            acc_n = 0
            for index, item in enumerate(boolq_question_list):
                instruction = self.instruction_template.format(passage=item['passage'], question=item['question'])
                result, rank = llm_chat.chat(instruction=instruction, history=[])
                if result:
                    answer = result[1]
                else:
                    answer = None
                try:
                    if rank == 0:
                        logger.info(instruction)
                        logger.info("correct: %s, answer : %s", str(item['answer'])[0], answer)
                        subject_result[str(index)] = answer
                        if subject_result[str(index)] == str(item['answer'])[0]:
                            acc_n += 1
                except Exception as e:
                    if rank == 0:
                        logger.info(e)
                    subject_result[str(index)] = str(e) + ". AI answer:" + answer
            if rank == 0:
                logger.info("Boolq dataset acc = %d/%d=%e", acc_n, len(boolq_question_list),
                             acc_n / len(boolq_question_list))
                total_n += len(boolq_question_list)
                total_acc_n += acc_n
                answer_result['Boolq_dataset'] = subject_result
                score_datas.append(['Boolq_dataset', len(boolq_question_list), acc_n / len(boolq_question_list)])
        if rank == 0:
            logger.info("Boolq acc = %d/%d=%e", total_acc_n, total_n, total_acc_n / total_n)
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
