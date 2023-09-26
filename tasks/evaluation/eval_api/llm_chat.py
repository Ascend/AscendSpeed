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


class LlmChat(metaclass=ABCMeta):
    @abstractmethod
    def chat(self, instruction: str, history: list) -> str:
        """
        大语言模型对话接口，输入指令和历史对话, 给出新的模型输出
        :param instruction: 当前指令
        :param history: 历史对话
        :return: 模型输出
        """
        raise NotImplementedError
