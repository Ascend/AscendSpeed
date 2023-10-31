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

"""Dataloaders."""


import torch
import random

from transformers import DataCollatorForSeq2Seq

from ascendspeed import get_args, get_tokenizer
from ascendspeed.core import parallel_state
from ascendspeed.error_utils import check_divisible, ensure_valid



def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # ascendspeed sampler
    if args.dataloader_type == 'single':
        if args.optimized_pipeline:
            batch_sampler = DynamicMicroBatchPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size())
        else:
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size())
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    tokenizer = get_tokenizer().tokenizer

    if  args.is_instruction_dataset:
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=32 if args.variable_seq_lengths else args.seq_length,
            return_tensors='pt',
            padding=True
        )
    else:
        collator = None

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       generator=torch.Generator().manual_seed(args.seed),
                                       collate_fn=collator,
                                       pin_memory=True)

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        ensure_valid(self.total_samples > 0, error_message='no sample' \
                                             ' to consume: {}'.format(self.total_samples))
        ensure_valid(self.consumed_samples < self.total_samples, error_message='no samples' \
                            ' left to consume: {}, {}'.format(self.consumed_samples, self.total_samples))
        ensure_valid(self.micro_batch_size > 0)
        ensure_valid(data_parallel_size > 0)
        ensure_valid(self.data_parallel_rank < data_parallel_size, error_message='data_parallel_rank' \
                     ' should be smaller than data size: {}, {}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        ensure_valid(self.total_samples > 0, error_message='no sample' \
                                             ' to consume: {}'.format(self.total_samples))
        ensure_valid(self.micro_batch_size > 0)
        ensure_valid(data_parallel_size > 0)
        ensure_valid(self.data_parallel_rank < data_parallel_size, error_message='data_parallel_rank' \
                     ' should be smaller than data size: {}, {}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        check_divisible(current_epoch_samples, self.micro_batch_times_data_parallel_size)

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                       * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size
        
        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class DynamicMicroBatchPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):

        args = get_args()
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.drop_last = drop_last
        self.dynamic_micro_batch_size = args.manual_mbs
        self.micro_batch_times_data_parallel_size = [
            self.dynamic_micro_batch_size[i] * data_parallel_size \
            for i in range(len(self.dynamic_micro_batch_size))
        ]

        # Sanity checks.
        ensure_valid(self.total_samples > 0, error_message='no sample' \
                                             ' to consume: {}'.format(self.total_samples))
        ensure_valid(self.consumed_samples < self.total_samples, error_message='no samples' \
                            ' left to consume: {}, {}'.format(self.consumed_samples, self.total_samples))
        ensure_valid(self.micro_batch_size > 0)
        ensure_valid(data_parallel_size > 0)
        ensure_valid(self.data_parallel_rank < data_parallel_size, error_message='data_parallel_rank' \
                     ' should be smaller than data size: {}, {}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self, n_mbs):
        start_idx = self.data_parallel_rank * self.dynamic_micro_batch_size[n_mbs]
        end_idx = start_idx + self.dynamic_micro_batch_size[n_mbs]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        n_mbs = 0
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size[n_mbs]:
                start_idx, end_idx = self.get_start_end_idx(n_mbs)
                yield batch[start_idx:end_idx]
                batch = []
                n_mbs = (n_mbs + 1) % len(self.micro_batch_times_data_parallel_size)

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]