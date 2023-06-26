# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Zero-shot datasets."""

import json
import math
import random

import numpy as np
import torch

from ascendspeed import get_args
from ascendspeed import print_rank_0
from ascendspeed import get_tokenizer
from .detokenizer import get_detokenizer


def build_dataset(task):
    """Helper function to select and build dataset."""

    if task == 'LAMBADA':
        return _build_lambada_dataset()
    if task == 'WIKITEXT103':
        return _build_wikitext103_dataset()
    if task == 'BoolQ':
        return _build_boolq_dataset()
    if task == 'PIQA':
        return _build_piqa_dataset()
    if task == 'HellaSwag':
        return _build_hellaswag_dataset()
    raise NotImplementedError('dataset for {} task is not '
                              'implemented.'.format(task))


class _LMDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, seq_len, pad_idx, num_original_tokens,
                 num_tokenized_tokens, overalapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overalapping_eval = overalapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overalapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overalapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        pad_mask = [1] * num_tokens
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])
        if self.overalapping_eval != self.seq_len and idx != 0:
            pad_mask[:-self.overalapping_eval] *= 0

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


class _LambadaDataset(torch.utils.data.Dataset):

    def __init__(self, path, pad_idx, tokenizer, seq_len, strict=False):
        print_rank_0('> building lambada dataset from {} ...'.format(path))
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.tokenize(text)
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.tokenize(text[:start_idx].strip())
        last_token = self.tokenizer.tokenize(' ' + last_token)
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        num_tokens = len(tokens)
        pad_mask = [0] * num_tokens
        labels = self.labels[idx]
        pad_mask += [1] * len(labels)
        tokens = tokens + labels
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            pad_mask += [0] * (num_pad)
            tokens += [self.pad_idx] * num_pad
        pad_mask = np.array(pad_mask[1:])

        return {'text': np.array(tokens), 'pad_mask': pad_mask}


def _build_lambada_dataset():
    """Build lambada dataset."""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.valid_data) == 1
    val_dataset = _LambadaDataset(args.valid_data[0], tokenizer.eod, tokenizer,
                                  args.seq_length, args.strict_lambada)
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))

    return val_dataset


def _build_wikitext103_dataset():
    """"""
    args = get_args()
    tokenizer = get_tokenizer()

    assert len(args.valid_data) == 1
    with open(args.valid_data[0], "rb") as reader:
        entire_data = reader.read().decode('utf-8')
    num_original_tokens = len(entire_data.strip().split(" "))
    entire_data = get_detokenizer(args.valid_data[0])(entire_data)
    tokenized_data = tokenizer.tokenize(entire_data)
    num_tokenized_tokens = len(tokenized_data)

    val_dataset = _LMDataset(tokenized_data, args.seq_length, tokenizer.eod,
                             num_original_tokens, num_tokenized_tokens,
                             args.overlapping_eval)
    print_rank_0(' > number of original tokens: {}, number of detokenized '
                 'tokens: {}'.format(num_original_tokens, num_tokenized_tokens))

    return val_dataset


class BaseTask(torch.utils.data.Dataset):
    def __init__(self, max_data_num=None, temp_index=0):
        super().__init__()
        self.temp_index = temp_index
        self.examples = []
        self.max_data_num = max_data_num
        self.templates = self.templates_set_without_newline()
       
    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def preprocess_dataset(self):
        for example in self.dataset:
            example = self.preprocess_example(example)
            if example[0] is None:
                continue
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return input_str, output_str, label

    def __iter__(self):
        for input_str, output_str, label in self.examples:
            yield input_str, output_str, label


class _BoolQDataset(BaseTask):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                self.dataset.append(data)
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return [
            ("{passage}\nQuestion: {question}?\nAnswer:", " {answer}", ["No", "Yes"]),
            ("Passage: {passage} After reading this passage, I have a question: {question}? True or False? Answer:", " {answer}", ["False", "True"]),
            ("{passage} Based on the above text, what's the best answer to this question: {question}?", " {answer}", ["No", "Yes"]),
            ("Based on the following passage, {question}? {passage} Please answer yes or no.", " {answer}", ["No", "Yes"]),
            ("Exercise: read the text and answer the question by True or False. Text: {passage} Question: {question}?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{passage}", example["passage"])
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = 1 if example["answer"] else 0
        return input_str, answer_str, label


class _PIQADataset(BaseTask):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = []
        self.dataset = []
        data_path, label_path = path[0], path[1]
        with open(label_path, 'r') as f:
            for line in f.readlines():
                label = str(line.strip())
                self.labels.append(int(label))

        with open(data_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                data = json.loads(line)
                data.update({"label": self.labels[i]})
                self.dataset.append(data)
        self.class_num = 2
        self.preprocess_dataset()

    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["goal"]
        answer_str = [' '+example["sol1"], ' '+example["sol2"]]
        label = example["label"]
        return input_str, answer_str, label

class _HellaSwagDataset(BaseTask):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                self.dataset.append(data)
        self.class_num = 4
        self.preprocess_dataset()
   
    def templates_set_without_newline(self):
        return None

    def preprocess_example(self, example):
        input_str = example["ctx"]
        answer_str = []
        for i in range(self.class_num):
            answer_str.append(' ' + example["endings"][i])
        label = int(example["label"])
        return input_str, answer_str, label

def _build_boolq_dataset():
    args = get_args()
    assert len(args.valid_data) == 1
    val_dataset = _BoolQDataset(args.valid_data[0])
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))
    return val_dataset


def _build_piqa_dataset():
    args = get_args()
    path = args.valid_data[0].split(",")
    val_dataset = _PIQADataset(path)
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))
    return val_dataset


def _build_hellaswag_dataset():
    args = get_args()
    val_dataset = _HellaSwagDataset(args.valid_data[0])
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))
    return val_dataset
