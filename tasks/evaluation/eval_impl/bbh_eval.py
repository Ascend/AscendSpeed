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
from tasks.evaluation.eval_impl.template import BBH_TEMPLATE_DIR
from ascendspeed.error_utils import check_divisible_by_zero
logger = logging.getLogger(__name__)


class BBHEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{fewshot_template}Q: {question}\nA:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template

    def eval(self, llm_chat: LlmChat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        with open(BBH_TEMPLATE_DIR, encoding='utf-8') as f:
            bbh_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                bbh_dataset = json.load(f)
            subject_name = file[0: -5]
            subject_result = {}
            sample_n += len(bbh_dataset['examples'])
            acc_n = 0
            for idx, item in enumerate(bbh_dataset['examples']):
                instruction = self.instruction_template.format(fewshot_template=bbh_template[subject_name],
                                                               question=item['input'])
                chat_result, rank = llm_chat.chat(instruction=instruction, history=[])
                answer = None
                if chat_result:
                    answer = chat_result[0]
                try:
                    if rank == 0:
                        logger.info("correct: %s, AI: %S", item['target'], answer.splitlines()[0])
                        subject_result[str(idx)] = answer.splitlines()[0]
                        if subject_result[str(idx)] == item['target']:
                            acc_n += 1
                except Exception as e:
                    subject_result[str(idx)] = str(e) + f". AI answer: {answer}"
            if rank == 0:
                logging.info("%s acc = %d/%d=%e", subject_name, acc_n, len(bbh_dataset['examples']), check_divisible_by_zero(acc_n, len(bbh_dataset['examples'])))
                total_n += len(bbh_dataset['examples'])
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(bbh_dataset['examples']), check_divisible_by_zero(acc_n, len(bbh_dataset['examples']))])
        if rank == 0:
            logger.info("bbh acc = %d/%d=%e", total_acc_n, total_n, check_divisible_by_zero(total_acc_n, total_n))
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
