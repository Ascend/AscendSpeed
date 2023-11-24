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
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.template import AGIEVAL_TEMPLATE_DIR
from ascendspeed.error_utils import check_divisible_by_zero
logger = logging.getLogger(__name__)


class AGIEvalExam(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{fewshot_template}   {question}\n{question_template}\n{options}"
                                      "\n{answer_template}"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        with open(AGIEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            AGI_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                agi_question_list = []
                for line in f.readlines():
                    agi_question_list.append(json.loads(line))
            subject_name = file[0: -6]
            subject_result = {}
            sample_n += len(agi_question_list)
            acc_n = 0
            for idx, item in enumerate(agi_question_list):
                if item['passage']:
                    question = item['passage'] + '\n' + item['question']
                else:
                    question = item['question']
                if item['options']:
                    options = '\n'.join(item['options'])
                else:
                    options = ""
                if item['label']:
                    if isinstance(item['label'], list):
                        correct = ','.join(item['label'])
                    else:
                        correct = item['label']
                else:
                    if item['answer']:
                        correct = item['answer'].replace('$', '')
                    else:
                        correct = None
                instruction = self.instruction_template.format(fewshot_template=AGI_few_shot_template[subject_name][0],
                                                               question=question,
                                                               question_template=AGI_few_shot_template[subject_name][1],
                                                               options=options,
                                                               answer_template=AGI_few_shot_template[subject_name][2])
                chat_result, rank = chat.chat(instruction=instruction, history=[])
                answer = None
                if chat_result:
                    answer = chat_result[0]
                try:
                    if rank == 0:
                        final_result = answer.splitlines()[0].replace('$', '').replace('(', '').replace(')', '')
                        logger.info("correct: %s, AI: %s", correct, final_result)
                        subject_result[str(idx)] = final_result
                        if subject_result[str(idx)] == correct:
                            acc_n += 1
                except Exception as e:
                    subject_result[str(idx)] = str(e) + f". AI answer: {answer}"
            if rank == 0:
                logger.info("%s acc = %d/%d=%e", subject_name, acc_n, len(agi_question_list), check_divisible_by_zero(acc_n, len(agi_question_list)))
                total_n += len(agi_question_list)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(agi_question_list), check_divisible_by_zero(acc_n, len(agi_question_list))])
        if rank == 0:
            logger.info("AGIEval acc = %d/%d=%e", total_acc_n, total_n, check_divisible_by_zero(total_acc_n, total_n))
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
