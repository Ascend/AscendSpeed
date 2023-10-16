import os
import time
import logging

import numpy as np
import torch
from deepspeed.accelerator import get_accelerator

from ascendspeed import is_rank_0, print_rank_0, get_args
from ascendspeed.tokenizer.tokenizer import build_tokenizer
from ascendspeed.core import parallel_state
from ascendspeed.data.blendable_dataset import BlendableDataset
from ascendspeed.data.dataset_utils import get_datasets_weights_and_num_samples, get_split_by_range_, \
    get_train_valid_test_split_
from ascendspeed.data.mtf_dataset import MTFDataset, get_packed_indexed_dataset
from ascendspeed.data.indexed_dataset import make_dataset as make_indexed_dataset

logger = logging.getLogger(__name__)


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup
):
    """Build train, valid, and test datasets."""

    args = get_args()

    tokenizer = build_tokenizer(args)
    pad_token = tokenizer.pad
    eos_token = tokenizer.eos
    
    # Single dataset.
    if len(data_prefix) == 1:
        all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
            data_prefix=data_prefix[0],
            data_impl=data_impl,
            splits_string=splits_string,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
            skip_warmup=skip_warmup
        )
    # Blending dataset.
    else:

        output = get_datasets_weights_and_num_samples(data_prefix=data_prefix, train_valid_test_num_samples=train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                data_prefix=prefixes[i],
                data_impl=data_impl,
                splits_string=splits_string,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seed=seed,
                skip_warmup=skip_warmup
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        all_train_datasets = BlendableDataset(train_datasets, weights) \
                            if train_datasets else None
        all_valid_datasets = BlendableDataset(valid_datasets, weights) \
                            if valid_datasets else None
        all_test_datasets = BlendableDataset(test_datasets, weights) \
                            if test_datasets else None

    return all_train_datasets, all_valid_datasets, all_test_datasets


def build_dataset_group(
    dataset_group_name,
    paths,
    weights,
    splits,
    data_impl,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup,
    train_valid_test
):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed in the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]

    # Single dataset.
    if len(paths) == 1:
        dataset = _build_single_datasets(
            data_prefix=paths[0],
            range_string=splits[0],
            data_impl=data_impl,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed,
            skip_warmup=skip_warmup,
            dataset_group_name=dataset_group_name,
            train_valid_test=train_valid_test
        )
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is of the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(
                data_prefix=prefixes[i],
                range_string=splits[i],
                data_impl=data_impl,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                train_valid_test_num_samples=datasets_train_valid_test_num_samples[i],
                seed=seed,
                skip_warmup=skip_warmup,
                dataset_group_name=dataset_group_name,
                train_valid_test=train_valid_test
            )

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets


def _build_single_datasets(
    data_prefix,
    range_string,
    data_impl,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup,
    dataset_group_name,
    train_valid_test
):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(
        data_prefix=data_prefix,
        data_impl=data_impl,
        skip_warmup=skip_warmup
    )

    total_num_of_documents = list(packed_indexed_dataset.values())[0].sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                data_impl=data_impl,
                skip_warmup=skip_warmup,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
    skip_warmup
):
    """Build train, valid, and test datasets."""

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix, data_impl=data_impl, skip_warmup=skip_warmup)

    total_num_of_documents = list(packed_indexed_dataset.values())[0].sizes.shape[0]
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = DecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                data_impl=data_impl,
                skip_warmup=skip_warmup,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class DecoderPackedMTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        data_impl,
        skip_warmup,
        documents,
        num_samples,
        seq_length: int,
        pad_token: int,
        eos_token: int,
        seed,
    ):
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, data_impl=data_impl, skip_warmup=skip_warmup, documents=documents)

        self.pad_token = pad_token
        self.seq_length = seq_length

        self.shuffle_index = _build_index_mappings(name=name, data_prefix=data_prefix, nb_documents=len(documents), mtf_dataset=self.mtf_dataset, num_samples=num_samples, seq_length=seq_length, seed=seed)
    
    def __len__(self):
        return len(self.shuffle_index)

    def __getitem__(self, idx):
        doc_idx = self.shuffle_index[idx]
        item = self.mtf_dataset[doc_idx]
        return {
            "input_ids": self._pad_token(item["input_ids"][:-1], self.pad_token, np.int64),
            "attention_mask": self._pad_token(item["attention_mask"][:-1], 0, np.int64),
            "labels": self._pad_token(item["labels"][1:], -100, np.int64),
        }
    
    def _pad_token(self, token, pad_value, dtype):
        padded_token = np.full((self.seq_length,), pad_value, dtype=dtype)
        token_length = len(token)
        if token_length <= self.seq_length:
            padded_token[:token_length] = token
        else:
            padded_token = token[:self.seq_length]
        return padded_token.astype(dtype)


def _build_index_mappings(
    name,
    data_prefix,
    nb_documents,
    mtf_dataset,
    num_samples: int,
    seq_length: int,
    seed,
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    shuffle_idx_filename = _filename + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if is_rank_0():
        if not os.path.isfile(shuffle_idx_filename):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            start_time = time.time()
            epoch = 0
            shuffle_idx = []
            while len(shuffle_idx) <= num_samples:
                new_document_ids = _build_shuffle_idx(nb_documents=nb_documents, np_rng=np_rng)
                # Generate a shuffling of the entire dataset
                shuffle_idx.extend(new_document_ids.tolist())
                epoch += 1

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = get_accelerator().LongTensor([1])
    torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))

    return shuffle_idx


def _build_shuffle_idx(nb_documents: int, np_rng):
    """Build the range [0, dataset_size) and shuffle."""
    dtype_ = np.int64

    result = np.arange(start=0, stop=nb_documents, step=1, dtype=dtype_)

    # in-place shuffling
    np_rng.shuffle(result)

    return result
