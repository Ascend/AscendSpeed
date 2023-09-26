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
from tasks.task_eval.eval_api.dataset_eval import DatasetEval
from tasks.task_eval.eval_api.llm_chat import LlmChat
from tasks.task_eval.eval_impl.template import CEVAL_TEMPLATE_DIR
logger = logging.getLogger(__name__)


class CEvalExam(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="{fewshot_template}\n\n问：{question}\n答："):
        self.test_dir = test_dir
        self.instruction_template = instruction_template

    def eval(self, llm_chat: LlmChat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        with open(CEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            ceval_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            data_df = pd.read_csv(file_path)
            subject_name = file[0: -8]
            subject_result = {}
            sample_n += len(data_df)
            count = 0
            acc_n = 0
            for idx, row in data_df.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = self.instruction_template.format(fewshot_template=ceval_few_shot_template[subject_name],
                                                               question=test_question)
                chat_result, rank = llm_chat.chat(instruction=instruction, history=[])
                answer = None
                if chat_result:
                    answer = chat_result[0]
                try:
                    if rank == 0:
                        count += 1
                        subject_result[str(idx)] = answer
                        if subject_result[str(idx)] == row['answer']:
                            acc_n += 1
                except Exception as e:
                    subject_result[str(idx)] = str(e) + f". AI answer: {answer}"
            if rank == 0:
                logger.info("%s acc = %d/%d=%e", subject_name, acc_n, len(data_df), acc_n / len(data_df))
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])
        if rank == 0:
            logger.info("ceval acc = %d/%d=%e", total_acc_n, total_n, total_acc_n / total_n)
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        logger.info(score_df)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
