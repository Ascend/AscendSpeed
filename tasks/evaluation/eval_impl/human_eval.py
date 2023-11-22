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

import json
import os
import logging
import stat
import re
import sys
import subprocess
from typing import Iterable, Dict
import pandas as pd
from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.template import CODE_TEST_LOG_DIR
from ascendspeed.error_utils import check_divisible_by_zero
from ascendspeed.utils import WRITE_FILE_DEFAULT_FLAGS, WRITE_FILE_DEFAULT_MODES
logger = logging.getLogger(__name__)


def extract_answer_code(answer, task: dict):
    """
    :param answer:
    :param task:
    :return:
    """
    task_id = task['task_id']
    target_func = task['entry_point']
    test_case = task['test']
    save_file = f"{task_id}.py".replace("/", "-")
    code = answer
    code_lines = code.split("\n")
    target_func_flag = False
    if not os.path.exists(CODE_TEST_LOG_DIR):
        os.makedirs(CODE_TEST_LOG_DIR)
    test_code_path = "{}{}".format(CODE_TEST_LOG_DIR, save_file)
    with os.fdopen(os.open(test_code_path, WRITE_FILE_DEFAULT_FLAGS, WRITE_FILE_DEFAULT_MODES), 'w') as f:
        f.write("from typing import List\n")
        f.write("import math\n")
        for i, line in enumerate(code_lines):
            if i == 0 and line.lower() == "python":
                continue
            line_strip = line.strip()
            if len(line_strip) < 1:
                continue
            if line_strip[0] == line[0]:
                if line.startswith("from") or line.startswith("import"):
                    f.write(line)
                    f.write('\n')
                elif line.startswith("def"):
                    if re.split(r"\s+", line)[1] == target_func:
                        target_func_flag = True
                    f.write(line)
                    f.write('\n')
                else:
                    if target_func_flag:
                        break
            else:
                f.write(line)
                f.write('\n')
        f.write(test_case)
        f.write('\n')
        f.write(f'check({target_func})')
    return test_code_path


class HumanEval(DatasetEval):
    def __init__(self, test_dir,
                 instruction_template="The definition and function description of the python function are as follows. "
                                      "Please complete the implementation of the python function.\n{prompt}"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template

    def read_problems(self) -> Dict[str, Dict]:
        return {task["task_id"]: task for task in self.stream_jsonl(self.test_dir)}

    def stream_jsonl(self, test_dir: str) -> Iterable[Dict]:
        """
        Parses each jsonl line and yields it as a dictionary
        """
        for file in os.listdir(test_dir):
            test_code_path = os.path.join(self.test_dir, file)
            with os.fdopen(os.open(test_code_path, WRITE_FILE_DEFAULT_FLAGS, WRITE_FILE_DEFAULT_MODES)) as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        problems = self.read_problems()
        success_n = 0
        rank = None
        answer_result = {}
        for idx, (task_id, task) in enumerate(problems.items()):
            instruction = self.instruction_template.format(prompt=task['prompt'])
            chat_result, rank = chat.chat(instruction=instruction, history=[])
            answer = None
            if chat_result:
                answer = chat_result[0]
            try:
                if rank == 0:
                    python_execute = sys.executable
                    answer = task['prompt'] + ' ' + answer
                    logger.info('answer: ', answer)
                    test_file = extract_answer_code(answer, task)
                    result = subprocess.run([python_execute, test_file], capture_output=True, timeout=10)
                    if result.returncode != 0:
                        error_msg = result.stderr.decode("utf-8")
                        logger.info(error_msg)
                        answer_result[task_id] = error_msg
                    else:
                        answer_result[task_id] = 'passed'
                        success_n += 1
                        logger.info("%s : passed , acc : %s", task_id, check_divisible_by_zero(success_n, len(problems)))
            except Exception as e:
                if rank == 0:
                    logger.info("%s failed. %s", task_id, e)
            finally:
                pass
        if rank == 0:
            logger.info("acc = %s", {check_divisible_by_zero(success_n, len(problems))})
        return answer_result, None

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
