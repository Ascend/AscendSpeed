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

"""Processing data for pretraining and finetuning."""

import argparse
import json
import multiprocessing
import os
import sys
import time
import logging

import torch
try:
    import nltk
except ImportError:
    nltk = None


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from ascendspeed.tokenizer import build_tokenizer
from ascendspeed.data import indexed_dataset
from ascendspeed.data.data_handler import build_dataset, get_dataset_handler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars if nltk else object):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


def build_splitter(args):
    if nltk and args.split_sentences:
        nltk.download("punkt", quiet=True)
    if args.split_sentences:
        if not nltk:
            logger.error("NLTK is not available to split sentences.")
            raise Exception("nltk is not available")
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        if args.keep_newlines:
            # this prevents punkt from eating newlines after sentences
            final_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text=splitter._params,
                lang_vars=CustomLanguageVars())
        else:
            final_splitter = splitter

    else:
        final_splitter = IdentitySplitter()
    return final_splitter


def add_data_args(parser):
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str,
                       help='Path to input JSON or path or a huggingface dataset name')
    group.add_argument('--handler-name', type=str, default="",
                       help='specify a dataset handler')
    group.add_argument('--streaming', action='store_true',
                       help='weather to use streaming')
    group.add_argument('--hf-datasets-params', default=None,
                       help='huggingface load_dataset params')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')


def add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach.'
                            'This value must be greater than the initial size of the tokenizer.'
                            ' If this argument is used the value of `make-vocab-size-divisible-by` '
                            'will be ignored.')


def add_output_args(parser):
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')


def get_args():
    parser = argparse.ArgumentParser()

    add_data_args(parser)
    add_tokenizer_args(parser)
    add_output_args(parser)

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            logger.warning("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()

    tokenizer = build_tokenizer(args)
    splitter = build_splitter(args)

    logger.info("building dataset: %s", args.input)
    raw_data = build_dataset(args)

    handler = get_dataset_handler(args, raw_data, tokenizer, splitter)

    # serialize to bin&idx
    handler.serialize_to_disk()


if __name__ == '__main__':
    main()
