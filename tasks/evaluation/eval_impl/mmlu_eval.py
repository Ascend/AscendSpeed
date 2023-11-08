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
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.template import MMLU_TEMPLATE_DIR
logger = logging.getLogger(__name__)


class MmluEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{few_shot_examples}\n\n"
                                      "{question}\nAnswer:",
                 output_template1=r".*(?P<answer>[A|B|C|D])\..*",
                 output_template2=r"(?P<answer>[A|B|C|D])"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = [output_template1, output_template2]

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        with open(MMLU_TEMPLATE_DIR, encoding='utf-8') as f:
            mmlu_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            subject_name = file[0: -9]  # 文件命名规则是  {subject}_test.csv
            subject = subject_name.replace("_", " ")
            subject_result = {}
            acc_n = 0
            for idx, row in data_df.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = self.instruction_template.format(few_shot_examples=mmlu_few_shot_template[subject_name],
                                                               subject=subject,
                                                               question=test_question)
                chat_result, rank = chat.chat(instruction=instruction, history=[])
                answer = None
                if chat_result:
                    answer = chat_result[0]
                try:
                    if rank == 0:
                        logger.info(instruction)
                        match_flag = False
                        for template in self.output_template:
                            try:
                                result = re.match(template, answer)
                                logger.info(
                                    "correct: %s, answer : %s, AI: %s", row['answer'], result.group('answer'), answer)
                                subject_result[str(idx)] = result.group("answer")
                                if subject_result[str(idx)] == row['answer']:
                                    acc_n += 1
                                match_flag = True
                                break
                            except Exception as e:
                                logger.info(e)
                                continue
                        if not match_flag:
                            logger.info("xx. AI answer: %s", answer)
                except Exception as e:
                    if rank == 0:
                        logger.info(e)
                    subject_result[str(idx)] = str(e) + ". AI answer:" + answer
            if rank == 0:
                logger.info("%s acc = %d/%d=%e", subject_name, acc_n, len(data_df), acc_n / len(data_df))
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])
        if rank == 0:
            logger.info("MMLU acc = %d/%d=%e", total_acc_n, total_n, total_acc_n / total_n)
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
