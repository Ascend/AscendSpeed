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

"""Convert weight when model parallel change"""

import argparse
import os

import torch

from ckpt_utils import column_split
from ckpt_utils import make_ascendspeed_model_dirs
from ckpt_utils import pad_embed
from ckpt_utils import permute_qkv_weight
from ckpt_utils import print_model
from ckpt_utils import row_split
from ckpt_utils import save_ascendspeed_model


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir",
                        help="source model dir of ascendspeed llama weight")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir",
                        help="target model dir of ascendspeed llama weight")
    parser.add_argument("--orig-vocab-size", type=int, default=32000, help="origin vocabulary size")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128,
                        help="should be consistent with ascendspeed")
    parser.add_argument("--src-tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--src-pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--tgt-tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--tgt-pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--type", type=str, choices=["7B", "13B", "30B", "65B"], default="7B")

    return parser.parse_args()


model_config = {
    "7B": [32, 4096, 32],
    "13B": [40, 5120, 40],
    "30B": [60, 6656, 52],
    "65B": [80, 8192, 64]
}

entire_model = {}


def check_model_dir(model_dir, tp_size, pp_size):
    m_dirs = os.listdir(model_dir)
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            if pp_size == 1:
                rank = 'mp_rank_{:02d}'.format(tp_rank)
            else:
                rank = 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank)
            assert rank in m_dirs, f"{rank} should be in {model_dir}"
    assert len(m_dirs) == tp_size * pp_size, f"there should be {tp_size * pp_size} subdirectory in {model_dir}"


def get_weight_from_name(k):
    global entire_model
    return entire_model[k]


def merge_weight(entire_model_dic, tp_models, k, pp_i, tot_i, dim):
    ws = [tm[k.format(pp_i)] for tm in tp_models]
    entire_model_dic[k.format(tot_i)] = torch.cat(ws, dim=dim)


def merge_pp_tp_models(pp_size, tp_size, input_model_dir, orig_vocab_size, num_heads, num_layer, hid_size):
    global entire_model
    assert num_heads % tp_size == 0, "num_head must be divisible by tensor model parallel"
    assert num_layer % pp_size == 0, "num_layers must be divisible by pipeline model parallel"
    pp_n_layer = num_layer // pp_size

    for pp_rank in range(pp_size):
        tp_models = []
        offset = pp_rank * pp_n_layer
        for tp_rank in range(tp_size):
            if pp_size == 1:
                model_dir = os.path.join(input_model_dir, 'mp_rank_{:02d}'.format(tp_rank))
            else:
                model_dir = os.path.join(input_model_dir, 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank))
            sd = torch.load(os.path.join(model_dir, "model_optim_rng.pt"), map_location=torch.device('cpu'))['model']
            tp_models.append(sd)

        if pp_rank == 0:
            emb_ws = [tm["embedding.word_embeddings.weight"] for tm in tp_models]
            entire_model["embedding.word_embeddings.weight"] = torch.cat(emb_ws, dim=0)[:orig_vocab_size, ...].clone()

        if pp_rank == pp_size - 1:
            entire_model["language_model.final_layernorm.weight"] = tp_models[0][
                "language_model.final_layernorm.weight"]
            head_ws = [tp_models[tp_rank]["lm_head.lm_head.weight"] for tp_rank in range(tp_size)]
            entire_model["lm_head.lm_head.weight"] = torch.cat(head_ws, dim=0)[:orig_vocab_size, ...].clone()

        for pp_i in range(pp_n_layer):
            g_i = offset + pp_i
            entire_model[f"language_model.layers.{g_i}.attention.rotary_emb.inv_freq"] = tp_models[0][
                f"language_model.layers.{pp_i}.attention.rotary_emb.inv_freq"]
            entire_model[f"language_model.layers.{g_i}.input_layernorm.weight"] = tp_models[0][
                f"language_model.layers.{pp_i}.input_layernorm.weight"]
            entire_model[f"language_model.layers.{g_i}.post_attention_layernorm.weight"] = tp_models[0][
                f"language_model.layers.{pp_i}.post_attention_layernorm.weight"]

            # qkv split
            qkv_key = "language_model.layers.{}.attention.query_key_value.weight"
            qkv_len = tp_models[0][qkv_key.format(pp_i)].shape[0]
            assert qkv_len % 3 == 0, "qkv weight should be divisible by 3"
            s1, s2 = qkv_len // 3, qkv_len // 3 * 2

            qs = [permute_qkv_weight(tm[qkv_key.format(pp_i)], num_heads, hid_size, tp_size, split=True)[:s1,
                  ...].clone() for tm in tp_models]
            ks = [permute_qkv_weight(tm[qkv_key.format(pp_i)], num_heads, hid_size, tp_size, split=True)[s1:s2,
                  ...].clone() for tm in tp_models]
            vs = [permute_qkv_weight(tm[qkv_key.format(pp_i)], num_heads, hid_size, tp_size, split=True)[s2:,
                  ...].clone() for tm in tp_models]

            entire_model[qkv_key.format(g_i) + "_query"] = torch.cat(qs, dim=0)
            entire_model[qkv_key.format(g_i) + "_key"] = torch.cat(ks, dim=0)
            entire_model[qkv_key.format(g_i) + "_value"] = torch.cat(vs, dim=0)
            merge_weight(entire_model, tp_models, "language_model.layers.{}.attention.dense.weight", pp_i, g_i, dim=1)
            merge_weight(entire_model, tp_models, "language_model.layers.{}.mlp.gate_proj.weight", pp_i, g_i, dim=0)
            merge_weight(entire_model, tp_models, "language_model.layers.{}.mlp.up_proj.weight", pp_i, g_i, dim=0)
            merge_weight(entire_model, tp_models, "language_model.layers.{}.mlp.down_proj.weight", pp_i, g_i, dim=1)
    return entire_model


def generate_ascendspeed_weights(tp_size, pp_size, output_model_dir, make_vocab_size_divisible_by, num_heads, num_layer,
                                 hid_size):
    release_model_dir = os.path.join(output_model_dir, "release")
    assert num_heads % tp_size == 0, "num_head must be divisible by tensor model parallel"
    assert num_layer % pp_size == 0, "num_layers must be divisible by pipeline model parallel"
    pp_n_layer = num_layer // pp_size

    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            model_dic = {"checkpoint_version": 3.0}
            rank_model = {}

            emb_w = get_weight_from_name("embedding.word_embeddings.weight")
            emb_w = pad_embed(emb_w, make_vocab_size_divisible_by, tp_size)

            if pp_rank == 0:
                rank_model["embedding.word_embeddings.weight"] = row_split(emb_w, tp_size, tp_rank)

            if pp_rank == pp_size - 1:
                rank_model["language_model.final_layernorm.weight"] = get_weight_from_name(
                    "language_model.final_layernorm.weight").clone()
                rank_model["lm_head.lm_head.weight"] = row_split(
                    pad_embed(get_weight_from_name("lm_head.lm_head.weight"), make_vocab_size_divisible_by, tp_size),
                    tp_size,
                    tp_rank)

            for pp_i in range(pp_n_layer):
                g_i = pp_n_layer * pp_rank + pp_i
                rank_model[f"language_model.layers.{pp_i}.attention.rotary_emb.inv_freq"] = get_weight_from_name(
                    f"language_model.layers.{g_i}.attention.rotary_emb.inv_freq")
                qkv_key = f"language_model.layers.{g_i}.attention.query_key_value.weight"
                qw = row_split(get_weight_from_name(qkv_key + "_query"), tp_size, tp_rank)
                kw = row_split(get_weight_from_name(qkv_key + "_key"), tp_size, tp_rank)
                vw = row_split(get_weight_from_name(qkv_key + "_value"), tp_size, tp_rank)
                permute_w = permute_qkv_weight(torch.cat([qw, kw, vw], dim=0), num_heads, hid_size, tp_size)
                rank_model[f"language_model.layers.{pp_i}.attention.query_key_value.weight"] = permute_w

                rank_model[f"language_model.layers.{pp_i}.attention.dense.weight"] = column_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.attention.dense.weight"), tp_size, tp_rank)

                rank_model[f"language_model.layers.{pp_i}.mlp.gate_proj.weight"] = row_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.gate_proj.weight"), tp_size, tp_rank)
                rank_model[f"language_model.layers.{pp_i}.mlp.up_proj.weight"] = row_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.up_proj.weight"), tp_size, tp_rank)
                rank_model[f"language_model.layers.{pp_i}.mlp.down_proj.weight"] = column_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.down_proj.weight"), tp_size, tp_rank)

                rank_model[f"language_model.layers.{pp_i}.input_layernorm.weight"] = get_weight_from_name(
                    f"language_model.layers.{g_i}.input_layernorm.weight").clone()
                rank_model[f"language_model.layers.{pp_i}.post_attention_layernorm.weight"] = get_weight_from_name(
                    f"language_model.layers.{g_i}.post_attention_layernorm.weight").clone()
            if tp_rank == 0 and pp_rank == 0:
                print_model(rank_model)

            save_ascendspeed_model(model_dic, rank_model, pp_size, tp_rank, pp_rank, release_model_dir)


def print_result(arg):
    print("=" * 100)
    print(
        f"weight converted from (tp={arg.tgt_tensor_model_parallel_size},pp={arg.tgt_pipeline_model_parallel_size})"
        f"to (tp={arg.tgt_tensor_model_parallel_size},pp={arg.tgt_pipeline_model_parallel_size}) success.."
        f"\nthe converted weights are stored in {arg.output_model_dir}"
    )
    print("=" * 100)


if __name__ == '__main__':
    args = get_args()
    n_layer, hidden_size, n_heads = model_config[args.type]

    check_model_dir(args.input_model_dir, args.src_tensor_model_parallel_size, args.src_pipeline_model_parallel_size)
    make_ascendspeed_model_dirs(args.output_model_dir)

    merge_pp_tp_models(args.src_pipeline_model_parallel_size, args.src_tensor_model_parallel_size,
                       args.input_model_dir, args.orig_vocab_size, n_heads, n_layer, hidden_size)
    generate_ascendspeed_weights(args.tgt_tensor_model_parallel_size, args.tgt_pipeline_model_parallel_size,
                                 args.output_model_dir, args.make_vocab_size_divisible_by, n_heads, n_layer,
                                 hidden_size)
    print_result(args)
