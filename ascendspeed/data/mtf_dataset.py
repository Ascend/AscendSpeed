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

"""Multitask Finetune style dataset."""

import time
import glob
import re

import numpy as np
import torch

from ascendspeed import print_rank_0
from ascendspeed.data.indexed_dataset import make_dataset as make_indexed_dataset

class MTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        data_impl,
        skip_warmup,
        documents,
    ):
        # Params to store.
        self.name = name

        # Dataset.
        self.packed_indexed_dataset = get_packed_indexed_dataset(data_prefix, data_impl=data_impl, skip_warmup=skip_warmup)

        # Checks
        assert np.min(documents) >= 0
        assert len(self.packed_indexed_dataset) > 0

        self.length = list(self.packed_indexed_dataset.values())[0].sizes.shape[0]

        assert np.max(documents) < self.length
        for dataset in self.packed_indexed_dataset.values():
            assert dataset.sizes.shape[0] == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        packed_data = dict()
        for key, dataset in self.packed_indexed_dataset.items():
            packed_data[key] = dataset.get(idx)
            assert len(packed_data[key]) > 0
        return packed_data


def get_packed_indexed_dataset(data_prefix: str, data_impl: str, skip_warmup: bool):
    index_dataset_name = f"{data_prefix}_packed_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_packed_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()
    for field in all_field:
        packed_dataset[field] = get_indexed_dataset_(
            f"{data_prefix}_packed_{field}_document", data_impl, skip_warmup
        )
    return packed_dataset


def get_indexed_dataset_(path, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(path,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset
