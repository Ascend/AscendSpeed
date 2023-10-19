# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA.
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



import math
import os
import time
import collections

import numpy as np
import torch
from deepspeed.accelerator import get_accelerator

from ascendspeed import (
    get_args,
    print_rank_0
)
from ascendspeed.core import parallel_state
from ascendspeed.data.indexed_dataset import make_dataset as make_indexed_dataset
from ascendspeed.error_utils import check_divisible, check_equal


DSET_TYPE_BERT = 'standard_bert'
DSET_TYPE_ICT = 'ict'
DSET_TYPE_T5  = 't5'

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5]


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    check_divisible(len(data_prefix), 2)
    num_datasets = len(data_prefix) // 2
    weights = [0]*num_datasets
    prefixes = [0]*num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2*i])
        prefixes[i] = (data_prefix[2*i+1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [int(math.ceil(val * weight * 1.005))
             for val in train_valid_test_num_samples])


    return prefixes, weights, datasets_train_valid_test_num_samples


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    #print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary, None)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    check_equal(len(tokentypes), num_tokens)
    check_equal(len(masked_positions, len(masked_labels)))

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    check_equal(indexed_dataset.sizes.shape[0], indexed_dataset.doc_idx[-1])
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    print_rank_0(' > indexed dataset stats:')
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.doc_idx.shape[0] - 1))
    print_rank_0('    number of sentences: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_split_by_range_(range_string, size):
    """ Get dataset splits based on a range:
    range_string is in the form  START%:END%  for e.g. 0.2:0.8
    outputs an array of two values [start_index, end_index]
    """
    # some checks that range is given in the correct form
    splits = [float(i) for i in range_string.split(":")]
    check_equal(len(splits), 2, "splits should be passed as start:end")
    assert splits[0] <= 1 and splits[1] <= 1
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits_index = [round(s * float(size)) for s in splits]
    check_equal(len(splits_index), 2)
    return splits_index


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    check_equal(len(splits_index), 4)
    check_equal(splits_index[-1], size)
    return splits_index

def get_samples_mapping(indexed_dataset,
                        data_prefix,
                        num_epochs,
                        max_num_samples,
                        max_seq_length,
                        short_seq_prob,
                        seed,
                        name,
                        binary_head):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""
    args = get_args()
    if args.train_data_exact_num_epochs is not None and name == 'train':
        num_epochs = args.train_data_exact_num_epochs
        max_num_samples = np.iinfo(np.int64).max - 1
    else:
        if not num_epochs:
            if not max_num_samples:
                raise ValueError("Need to specify either max_num_samples "
                                "or num_epochs")
            num_epochs = np.iinfo(np.int32).max - 1
        if not max_num_samples:
            max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if args.train_data_exact_num_epochs is not None and name == 'train':
        indexmap_filename += '_exact{}ep'.format(num_epochs)
    else:
        if num_epochs != (np.iinfo(np.int32).max - 1):
            indexmap_filename += '_{}ep'.format(num_epochs)
        if max_num_samples != (np.iinfo(np.int64).max - 1):
            indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    if name == 'train':
        # force to use certain index files
        if args.train_idx_path is not None:
            indexmap_filename = args.train_idx_path

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
       not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        check_equal(indexed_dataset.doc_idx.dtype, np.int64)
        check_equal(indexed_dataset.sizes.dtype, np.int32)

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(' > building sapmles index mapping for {} ...'.format(
            name))
        # First compile and then import.
        from megatron.data import helpers
        samples_mapping = helpers.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose,
            2 if binary_head else 1)
        print_rank_0(' > done building sapmles index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
                         time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    if get_accelerator().device_count() > 0: # Skip when CPU-only
        counts = get_accelerator().LongTensor([1])
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        assert counts[0].item() == (
            torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping
