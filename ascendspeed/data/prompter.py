# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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

import json
import os.path as osp
from dataclasses import dataclass


@dataclass
class AlpacaTemplate:
    system_token = ""
    user_token = "### Instruction:"
    assistant_token = "### Response:"
    end_token = ""
    system = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request. " \
             "Please note that you need to think through your response logically and step by step."


class Prompter(object):

    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.assistant_role = "assistant"
    
    def generate_training_prompt(self, messages) -> str:
        prompt = self.template.system_token + "\n" + self.template.system + self.template.end_token + "\n"

        for message in messages:
            if message["role"] == self.user_role:
                prompt += self.template.user_token + "\n" + message["content"] + self.template.end_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["content"] \
                + self.template.end_token + "\n"
        
        return prompt
