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

from abc import abstractmethod, ABCMeta
from tasks.evaluation.eval_api.llm_chat import LlmChat


class DatasetEval(metaclass=ABCMeta):
    @abstractmethod
    def eval(self, llm_chat: LlmChat):
        """
        数据集评估
        :param llm_chat: 大语言模型
        :return: 评估结果
        dict是各主题的答案, 结构 {"subject_name": {"题号": "选择", ”1“:"C",...}}
        pd.DataFrame是评估分数，包含字段：subject, question_n, acc
        """
        raise NotImplementedError

    @abstractmethod
    def top_k_eval(self, ):
        """
        数据集评估
        :param llm_chat: 大语言模型
        :return: 评估结果
        """
        raise NotImplementedError
