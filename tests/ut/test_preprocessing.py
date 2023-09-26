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

import io
import json
import re
import os
import sys
import unittest
import tempfile
import shutil
from unittest import mock
from pathlib import Path

import datasets
import numpy as np
from wrapt_timeout_decorator import timeout
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer


def write_jsonl(path, lines_num=1000, line_length=1024, is_pretrain=True):
    def get_text_dict(line_length):
        return {"text": "It's a wonderful world. I'm just walking on air. \
        Talk of heaven on earth. I've got more than my share. \Haven't got a care. \
        Happy all day through. It's a wonderful world. Loving wonderful you!"}

    def get_instruction_dict(line_length):
        return {
            "instruction": "Create a classification task by clustering the given list of items.",
            "input": "Apples, oranges, bananas, strawberries, pineapples",
            "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples"
        }

    with io.open(path, "w", encoding="utf-8") as f:

        for _ in range(lines_num):
            if is_pretrain:
                rec = get_text_dict(line_length)
            else:
                rec = get_instruction_dict(line_length)
            x = json.dumps(rec, indent=0, ensure_ascii=False)
            x = re.sub(r'\n', ' ', x, 0, re.M)
            f.write(x + "\n")


class ASTestPreprocessing(unittest.TestCase):
    """AscendSpeed preprocess test"""

    def setUp(self):
        ut_test_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(os.path.dirname(ut_test_dir))
        self.data_path = "/home/dataset"
        sys.path.append(self.root_dir)

        from tools.preprocess_data import main
        from ascendspeed.data.indexed_dataset import make_dataset
        from ascendspeed.data.prompter import AlpacaTemplate
        self.test_function = main
        self.make_data_function = make_dataset

        # tokenizer thing
        self.tokenizer_type = "GPT2BPETokenizer"
        self.merge_file_path = os.path.join(self.data_path, "gpt2-merges.txt")
        self.vocab_path = os.path.join(self.data_path, "gpt2-vocab.json")
        self.tokenizer_name_or_path = None

        # data thing
        self.data_impl = "mmap"
        self.template = AlpacaTemplate()

    @timeout(1200)
    def test_preprocess_pretrain_data(self):
        output_dir = tempfile.mkdtemp()

        input_path = f"{output_dir}/input.jsonl"
        write_jsonl(input_path)

        output_prefix = f"{output_dir}/test-ds"

        with mock.patch('argparse._sys.argv', [
            "preprocess_data.py",
            "--streaming",
            "--input", input_path,
            "--output-prefix", output_prefix,
            "--dataset-impl", self.data_impl,
            "--tokenizer-type", self.tokenizer_type,
            "--merge-file", self.merge_file_path,
            "--vocab", self.vocab_path,
            "--workers", '2'
        ]):
            self.test_function()

        # test bin and idx exist or not
        for ext in ["bin", "idx"]:
            tgt_path = f"{output_prefix}_text_document.{ext}"
            self.assertTrue(Path(tgt_path).exists(), )

        hf_ds = self._get_hf_dataset("json", input_path, is_instruction_data=False)
        pretrain_ds = self.make_data_function(f'{output_prefix}_text_document', self.data_impl)
        for i, _ in enumerate(hf_ds):
            # test content in bin and idx same with huggingface
            np.testing.assert_allclose(pretrain_ds[i], hf_ds[i]["input_ids"])

        shutil.rmtree(output_dir)

    @timeout(1200)
    def test_preprocess_instruction_data(self):
        output_dir = tempfile.mkdtemp()

        input_path = f"{output_dir}/input.jsonl"
        write_jsonl(input_path, is_pretrain=False)

        output_prefix = f"{output_dir}/test-ds"

        with mock.patch('argparse._sys.argv', [
            "preprocess_data.py",
            "--input", input_path,
            "--handler-name", "GeneralInstructionHandler",
            "--output-prefix", output_prefix,
            "--dataset-impl", self.data_impl,
            "--tokenizer-type", self.tokenizer_type,
            "--merge-file", self.merge_file_path,
            "--vocab", self.vocab_path,
            "--workers", '2'
        ]):
            self.test_function()

        for ext in ["bin", "idx"]:
            input_ids_path = f"{output_prefix}_packed_input_ids_document.{ext}"
            attention_mask_path = f"{output_prefix}_packed_attention_mask_document.{ext}"
            labels_path = f"{output_prefix}_packed_labels_document.{ext}"
            self.assertTrue(Path(input_ids_path).exists(), )
            self.assertTrue(Path(attention_mask_path).exists(), )
            self.assertTrue(Path(labels_path).exists(), )

        input_ids_data = self.make_data_function(f"{output_prefix}_packed_input_ids_document", self.data_impl)
        attention_mask_data = self.make_data_function(f"{output_prefix}_packed_attention_mask_document", self.data_impl)
        hf_ds = self._get_hf_dataset("json", input_path, template=self.template)
        for i, _ in enumerate(hf_ds):
            np.testing.assert_allclose(input_ids_data[i], hf_ds[i]["input_ids"])
            np.testing.assert_allclose(attention_mask_data[i], hf_ds[i]["attention_mask"])

        shutil.rmtree(output_dir)

    def _get_hf_dataset(self, data_format, data_path, json_key="text", template=None, is_instruction_data=True):
        if self.tokenizer_type == "GPT2BPETokenizer":
            tokenizer = GPT2Tokenizer(
                self.vocab_path,
                self.merge_file_path,
                errors='replace',
                special_tokens=[],
                max_len=None
            )
        elif self.tokenizer_type == "BertWordPieceCase":
            tokenizer = BertTokenizer(vocab_file=self.vocab_path, do_lower_case=False)
        elif self.tokenizer_type == "BertWordPieceLowerCase":
            tokenizer = BertTokenizer(vocab_file=self.vocab_path, do_lower_case=True)
        elif self.tokenizer_type == "PretrainedFromHF":
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=False)
        else:
            raise NotImplementedError
        rw_dataset = datasets.load_dataset(data_format,
                                           data_files=data_path,
                                           split="train",
                                           num_proc=4,
                                           cache_dir="./cache",
                                           streaming=False)

        def instruction_func(sample):
            return tokenizer(
                f"{template.system_token}\n{template.system}{template.end_token}\n{template.user_token}\n" +
                f"{sample['instruction']}\n{sample['input']}{template.end_token}\n{template.assistant_token}\n" +
                f"{sample['output']}{template.end_token}\n",
            )

        def pretrain_func(sample):
            return tokenizer(
                sample[json_key],
            )

        ds = rw_dataset.map(instruction_func if is_instruction_data else pretrain_func, num_proc=8)

        return ds
