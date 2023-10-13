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

import os
import sys
import time
import glob
import json
import logging

import torch
import numpy as np
from datasets import load_dataset

from ascendspeed.data import indexed_dataset
from ascendspeed.data.prompter import Prompter, AlpacaTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["get_dataset_handler", "build_dataset"]

DEFAULT_CACHE_DIR = "~/tmp"


class BaseDatasetHandler(object):
    """
    a base handler to tokenize or/and prompt your own dataset
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        self.args = args
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.raw_datasets = raw_datasets
        self.max_seq_len = args.seq_length
        self.tokenized_dataset = None

    @property
    def _unwrapped_tokenizer(self):
        """get huggingface tokenizer"""
        return self.tokenizer.tokenizer

    def get_tokenized_data(self):
        """get tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        proc_kwargs = {} if self.args.streaming else {"num_proc": self.args.workers}
        return self.raw_datasets.map(self._filter, remove_columns=remove_columns, **proc_kwargs)

    def serialize_to_disk(self):
        """save idx and bin to disk"""
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                         impl=self.args.dataset_impl,
                                                         vocab_size=None)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info("Time to startup:%s", startup_end - startup_start)

        for i, doc in enumerate(iter(self.tokenized_dataset), start=1):
            for key in self.args.json_keys:
                sentences = doc[key]
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    total_bytes_processed += len(sentence) * np.int32().itemsize
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                logger.info("Processed %s documents (%s docs/s, %s MB/s).", i, i / elapsed, mbs)

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented


class GeneralPretrainHandler(BaseDatasetHandler):
    """
    a general pretrain dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        if self._text_keys:
            self.args.json_keys = self._text_keys

    @property
    def _text_keys(self):
        return []

    def _pre_process(self, sample):
        return sample

    def _filter(self, sample):
        sample = self._pre_process(sample)
        for key in self.args.json_keys:
            text = sample[key]
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                if len(sentence) > 0:
                    sentence_ids = self._tokenize(sentence)
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
                doc_ids[-1]['attention_mask'].append(1)
                doc_ids[-1]['labels'].append(self.tokenizer.eod)
            sample[key] = doc_ids
            # for now, only input_ids are saved
            sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample


class AlpacaPretrainHandler(GeneralPretrainHandler):
    """
    alpaca-data-conversation pretrain dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
       
        self.message_format = "A chat between a curious user and an artificial intelligence assistant. " \
                              "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                              "USER: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
                              "### Instruction:\n{instruction}\n\n###{inputs}\n\n### Response: ASSISTANT: {response}"

    def _filter(self, sample):
        key = "text"
        text = self.message_format.format(
            instruction=sample.get("instruction"), 
            inputs=f" Input:\n{sample.get('input')}" if sample.get("input") else None,
            response=sample.get("output"))
        doc_ids = []
        for sentence in self.splitter.tokenize(text):
            if len(sentence) > 0:
                sentence_ids = self._tokenize(sentence)
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
        sample[key] = doc_ids
        sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample
    

class GeneralInstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = Prompter(AlpacaTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError
    
    def _is_muti_turn(self) -> bool:
        try:
            is_multi_turn = True if isinstance(self._human_prefix, str) else False
        except NotImplementedError:
            is_multi_turn = False
        return is_multi_turn

    def _format_msg(self, sample):
        """format sample info"""
        if not self.is_multi_turn:
            messages = [
                dict(
                    role=self.prompter.user_role,
                    content=sample[self._instruction_key] + "\n" + sample[self._input_key]),
                dict(role=self.prompter.assistant_role, content=sample[self._output_key])
            ]
            return messages
        
        messages = []
        turns = sample[self._instruction_key].split(self._human_prefix)

        for msg in turns:
            if not msg:
                continue
            tmp = msg.split(self._assistant_prefix)
            if len(tmp) > 1:
                messages.append(dict(role=self.prompter.user_role, content=tmp[0].strip()))
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[1].strip()))
            else:
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[0].strip()))
        messages.pop()
        messages.append(dict(role=self.prompter.assistant_role, content=sample[self._output_key].strip()))
        return messages

    def _filter(self, sample):
        messages = self._format_msg(sample)
        full_prompt = self.prompter.generate_training_prompt(messages)
        tokenized_full_prompt = self._tokenize(full_prompt)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        if not self.train_on_inputs:
            user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                self.prompter.template.assistant_token + "\n"
            tokenized_user_prompt = self._tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"][:user_prompt_len] = [self.ignored_label] * user_prompt_len

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


class BelleMultiTurnInstructionHandler(GeneralInstructionHandler):
    """
    BelleMultiTurn dataset handler
    """
    @property
    def _human_prefix(self) -> str:
        return "Human:"

    @property
    def _assistant_prefix(self) -> str:
        return "Assistant:"


class MOSSInstructionHandler(GeneralInstructionHandler):
    def _filter(self, sample):
        messages = []
        tokenized_chats = []

        for turn in sample["chat"].values():
            if not turn:
                continue

            user = turn["Human"].replace("<eoh>", "").replace("<|Human|>: ", "").strip()
            assistant = turn["MOSS"].replace("<|MOSS|>:", "").replace("<eom>", "").strip()

            messages.append(dict(role=self.prompter.user_role, content=user))
            messages.append(dict(role=self.prompter.assistant_role, content=assistant))

            full_prompt = self.prompter.generate_training_prompt(messages)
            tokenized_full_prompt = self._tokenize(full_prompt)

            if not self.train_on_inputs:
                user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                              self.prompter.template.assistant_token + "\n"
                tokenized_user_prompt = self._tokenize(user_prompt)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                             user_prompt_len:]

            tokenized_chats.append(tokenized_full_prompt)

        for key in self.args.json_keys:
            sample[key] = [chat[key] for chat in tokenized_chats]

        return sample


class LeetcodePythonInstructionHandler(GeneralInstructionHandler):
    @property
    def _instruction_key(self) -> str:
        return "code_with_problem"

    @property
    def _input_key(self) -> str:
        return "code_only"

    @property
    def _output_key(self) -> str:
        return "explanation_only"

    def _format_msg(self, sample):
        """format sample info"""
        messages = [
            dict(
                role=self.prompter.user_role,
                content=sample[self._instruction_key].split("```", maxsplit=1)[0].strip()),
            dict(
                role=self.prompter.assistant_role,
                content=sample[self._input_key] + "\n" + sample[self._output_key])
        ]
        return messages


class StackOverflowPythonPretrainHandler(GeneralPretrainHandler):
    @property
    def _text_keys(self):
        return ['text']

    def _pre_process(self, sample):
        sample['text'] = f"In python, {sample['title']}\n### Question:\n{sample['question_body']}\n" \
                         f"### Response:\n{sample['answer_body']}\n"


def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("curent module not found")
    handler = getattr(current_module, handler_name, None)
    if handler is None:
        handler = GeneralPretrainHandler
    logger.info("dataset will use %s to handle dataset", handler.__name__)
    return handler


def get_dataset_handler(args, raw_dataset, tokenizer, splitter):
    """
    get a handler instance
    """
    handler = _get_handler_cls(args.handler_name)

    handler_instance = handler(args, raw_dataset, tokenizer, splitter)
    return handler_instance


def _get_data_format(files):
    """get format with largest number"""
    all_support_format = {
        'parquet': 'parquet',
        'arrow': 'arrow',
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text'
    }
    format_num = {}
    for file in files:
        ext = file.split('.')[-1]
        format_num[ext] = format_num.get(ext, 0) + 1
    exts_with_num = sorted(format_num.items(), key=lambda x: x[1], reverse=True)
    has_data_file = False
    for ext, _ in exts_with_num:
        if ext in all_support_format:
            has_data_file = True
            break
    return (ext, all_support_format.get(ext)) if has_data_file else (None, None)


def _has_py_script(input_name):
    if os.path.isdir(input_name):
        dir_name = os.path.basename(input_name)
        if os.path.exists(os.path.join(input_name, dir_name + '.py')):
            has_py_script = True
        else:
            has_py_script = False
    else:
        if input_name.split('.')[-1] == 'py':
            has_py_script = True
        else:
            has_py_script = False
    return has_py_script


def build_dataset(args):
    """loading dataset by huggingface"""
    if args.handler_name == "MOSSInstructionHandler":
        # for MOSS, streaming is needed.
        args.streaming = True
    if args.hf_datasets_params:
        with open(args.hf_datasets_params, 'r') as fin:
            param_dict = json.load(fin)
        return load_dataset(**param_dict)
    cache_dir = DEFAULT_CACHE_DIR
    split_flag = "train"
    load_from_local = os.path.exists(args.input)
    if load_from_local:
        if _has_py_script(args.input):
            logger.info("loading data from a local python script")
            raw_datasets = load_dataset(
                args.input,
                split=split_flag,
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming
            )
        else:
            data_files = [args.input] if os.path.isfile(args.input) else \
                glob.glob(os.path.join(args.input, '*'))
            ext, data_format = _get_data_format(data_files)
            filtered_data_files = list(filter(lambda x: x.split('.')[-1] == ext, data_files))
            if filtered_data_files:
                logger.info("loading data from local file, format: %s," 
                            " file num: %s", data_format, len(data_files))
                raw_datasets = load_dataset(
                    data_format,
                    split=split_flag,
                    data_files=filtered_data_files,
                    num_proc=None if args.streaming else args.workers,
                    cache_dir=cache_dir,
                    streaming=args.streaming
                )
            else:
                raise Exception("unknown local data!")
    else:
        logger.info("loading data from remote huggingface")
        raw_datasets = load_dataset(
            args.input,
            split=split_flag,
            num_proc=None if args.streaming else args.workers,
            cache_dir=cache_dir,
            streaming=args.streaming
        )
    return raw_datasets
