# coding=utf-8
# Copyright Information:
# Huawei Technologies Co., Ltd. All Rights Reserved © 2020
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

"""Llama ckpt convert utils"""

import os
import stat

import logging
import torch

from ascendspeed.error_utils import check_divisible, check_equal
logging.basicConfig(level=logging.NOTSET)


def make_ascendspeed_model_dirs(output_model_dir, filename="latest_checkpointed_iteration.txt"):
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    flags = os.O_RDWR | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP
    with os.fdopen(os.open(os.path.join(output_model_dir, filename), flags, modes),
                   'w') as fout:
        fout.write("release\n")

    release_model_dir = os.path.join(output_model_dir, "release")
    os.makedirs(release_model_dir, exist_ok=True)


def row_split(w, tp, r):
    h = w.shape[0]
    check_divisible(h, tp)
    part_len = h // tp
    return w[r * part_len: (r + 1) * part_len, ...].clone()


def column_split(w, tp, r):
    dim1 = w.shape[1]
    check_divisible(dim1, tp)
    part_len = dim1 // tp
    return w[:, r * part_len: (r + 1) * part_len].clone()


def vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tp):
    # Pad vocab size so it is divisible by model parallel size and still having GPU friendly size.
    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tp
    while (after % multiple) != 0:
        after += 1
    return after


def pad_embed(w, make_vocab_size_divisible_by, tp, added_token_num):
    padded_size = vocab_size_with_padding(w.shape[0] + added_token_num, make_vocab_size_divisible_by, tp)
    if padded_size == w.shape[0]:
        return w.clone()
    return torch.cat([w, w[-(padded_size - w.shape[0]):, ...]], dim=0)


def permute_qkv_weight(w, n_head, hidden_size, tp, split=False):
    """ adapt for ascendspeed llama qkv layer """
    check_divisible(n_head, tp)
    check_divisible(hidden_size, n_head)
    np = n_head // tp
    hn = hidden_size // n_head
    w_s0, w_s1 = w.shape
    check_equal(w_s0, np * hn * 3)
    if not split:
        return w.reshape(3, np, hn, w.shape[1]).contiguous().permute(1, 0, 2, 3).reshape(w_s0,
                                                                                         w_s1).contiguous().clone()
    else:
        return w.reshape(np, 3, hn, w.shape[1]).contiguous().permute(1, 0, 2, 3).reshape(w_s0,
                                                                                         w_s1).contiguous().clone()


def permute_qkv_bias(bias, n_head, hidden_size, tp, split=False):
    """ adapt for ascendspeed llama qkv layer """
    check_divisible(n_head, tp)
    check_divisible(hidden_size, n_head)
    np = n_head // tp
    hn = hidden_size // n_head
    bias_shape = bias.shape[0]
    check_equal(bias_shape, np * hn * 3)
    if not split:
        return bias.reshape(3, np, hn).contiguous().permute(1, 0, 2).reshape(bias_shape).contiguous().clone()
    else:
        return bias.reshape(np, 3, hn).contiguous().permute(1, 0, 2).reshape(bias_shape).contiguous().clone()


def print_model(model):
    param_cnt = 0
    for p_name, v in model.items():
        if "check" in p_name:
            continue
        logging.info("p_name: %s, shape:%s, dtype: %s", p_name, str(v.shape), str(v.dtype))
        param_cnt += v.numel()
    logging.info("total param: %s", param_cnt)


def save_ascendspeed_model(config):
    model_dic = config.model_dic
    rank_model = config.rank_model
    pp_size = config.pp_size
    tp_rank = config.tp_rank
    pp_rank = config.pp_rank
    release_model_dir = config.release_model_dir

    model_dic['model'] = rank_model
    if pp_size == 1:
        model_dir = os.path.join(release_model_dir, 'mp_rank_{:02d}'.format(tp_rank))
    else:
        model_dir = os.path.join(release_model_dir, 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank))
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model_dic, os.path.join(model_dir, "model_optim_rng.pt"))
