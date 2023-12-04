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

import logging
import torch

from ckpt_utils import column_split
from ckpt_utils import make_ascendspeed_model_dirs
from ckpt_utils import pad_embed
from ckpt_utils import permute_qkv_weight
from ckpt_utils import print_model
from ckpt_utils import row_split
from ckpt_utils import save_ascendspeed_model

from ascendspeed.error_utils import check_equal, check_exist, check_divisible
from ascendspeed.data_classes import MergePpTpMergeConfig, MergeWeightConfig
from ascendspeed.data_classes import GenAscendWeightsConfig, SaveAscendspeedModelConfig
logging.basicConfig(level=logging.NOTSET)


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
    parser.add_argument("--added-token-num", type=int, default=0, help="the number of added tokens")
    parser.add_argument("--type", type=str,
                        help="There are four predefined types: [7B, 13B, 30B, 65B]")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="num layers", dest="num_layers")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="num heads", dest="num_heads")
    parser.add_argument("--num-kv-heads", type=int, default=None,
                        help="num kv heads", dest="num_kv_heads")
    parser.add_argument("--hidden-size", type=int, default=1,
                        help="hidden size", dest="hidden_size")
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--merge-mlp", action="store_true", default=False,
                        help="Merge gate and up mlp")

    return parser.parse_args()


model_config = {
    "7B": [32, 4096, 32],
    "13B": [40, 5120, 40],
    "30B": [60, 6656, 52],
    "65B": [80, 8192, 64]
}

args = get_args()
entire_model = {}


def check_model_dir(model_dir, tp_size, pp_size):
    m_dirs = os.listdir(model_dir)
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            if pp_size == 1:
                rank = 'mp_rank_{:02d}'.format(tp_rank)
            else:
                rank = 'mp_rank_{:02d}_{:03d}'.format(tp_rank, pp_rank)

            check_exist(rank, m_dirs)
    check_equal(len(m_dirs), tp_size * pp_size)


def get_weight_from_name(k):
    global entire_model
    if k in entire_model:
        return entire_model[k]
    else:
        raise KeyError(f"{k} is not in {entire_model.keys()}")


def merge_weight(config):
    ws = [tm["language_model"]["encoder"][config.src_k.format(config.pp_i)] for tm in config.tp_models]
    config.entire_model_dic[config.k.format(config.tot_i)] =\
        torch.cat(ws, dim=config.dim)


def merge_pp_tp_models(config):
    pp_size, tp_size = config.pp_size, config.tp_size
    input_model_dir = config.input_model_dir
    orig_vocab_size = config.orig_vocab_size
    num_layer, num_heads, hid_size = config.num_layer, config.num_heads, config.hid_size
    repeats = num_heads // args.num_kv_heads
    global entire_model
    check_divisible(num_heads, tp_size)
    check_divisible(num_layer, pp_size)
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
            emb_ws = [tm["language_model"]["embedding"]["word_embeddings"]["weight"] for tm in tp_models]
            entire_model["embedding.word_embeddings.weight"] = torch.cat(emb_ws, dim=0)[:orig_vocab_size, ...].clone()

        if pp_rank == pp_size - 1:
            entire_model["language_model.final_layernorm.weight"] = tp_models[0]["language_model"]["encoder"][
                "final_layernorm.weight"]
            head_ws = [tp_models[tp_rank]["language_model"]["output_layer"]["weight"] for tp_rank in range(tp_size)]
            entire_model["lm_head.lm_head.weight"] = torch.cat(head_ws, dim=0)[:orig_vocab_size, ...].clone()

        for pp_i in range(pp_n_layer):
            g_i = offset + pp_i
            entire_model[f"language_model.layers.{g_i}.attention.rotary_emb.inv_freq"] = \
            tp_models[0]["language_model"]["encoder"][f"layers.{pp_i}.self_attention.rotary_emb.inv_freq"]
            entire_model[f"language_model.layers.{g_i}.input_layernorm.weight"] = \
            tp_models[0]["language_model"]["encoder"][f"layers.{pp_i}.input_layernorm.weight"]
            entire_model[f"language_model.layers.{g_i}.post_attention_layernorm.weight"] = \
            tp_models[0]["language_model"]["encoder"][f"layers.{pp_i}.post_attention_layernorm.weight"]

            # qkv split
            qkv_key = "layers.{}.self_attention.query_key_value.weight"

            qkv_len = \
            tp_models[0]["language_model"]["encoder"][qkv_key.format(pp_i)].shape[0]

            check_divisible(qkv_len, repeats + 2)
            s1, s2 = qkv_len // (repeats + 2) * repeats, qkv_len // (repeats + 2) * (repeats + 1)

            qs = [permute_qkv_weight(tm["language_model"]["encoder"][qkv_key.format(pp_i)], (num_heads, hid_size, tp_size, args.num_kv_heads), split=True)[:s1,
                  ...].clone() for tm in tp_models]
            ks = [permute_qkv_weight(tm["language_model"]["encoder"][qkv_key.format(pp_i)], (num_heads, hid_size, tp_size, args.num_kv_heads), split=True)[s1:s2,
                  ...].clone() for tm in tp_models]
            vs = [permute_qkv_weight(tm["language_model"]["encoder"][qkv_key.format(pp_i)], (num_heads, hid_size, tp_size, args.num_kv_heads), split=True)[s2:,
                  ...].clone() for tm in tp_models]
            qkv_key_entire_model = "language_model.layers.{}.attention.query_key_value.weight"
            entire_model[qkv_key_entire_model.format(g_i) + "_query"] = torch.cat(qs, dim=0)
            entire_model[qkv_key_entire_model.format(g_i) + "_key"] = torch.cat(ks, dim=0)
            entire_model[qkv_key_entire_model.format(g_i) + "_value"] = torch.cat(vs, dim=0)

            merge_weight_config1 = MergeWeightConfig(entire_model, tp_models, "layers.{}.self_attention.dense.weight",
                                                     "language_model.layers.{}.attention.dense.weight",
                                                     pp_i, g_i, dim=1)
            merge_weight(merge_weight_config1)
            if args.merge_mlp:
                mlp_key = "layers.{}.mlp.".format(pp_i)
                mlp_len = tp_models[0][mlp_key + "proj.weight"].shape[0] // 2
                for tm in tp_models:
                    tm[mlp_key + "gate_proj.weight"] = tm["language_model"]["encoder"][mlp_key + "proj.weight"][
                                                       :mlp_len].clone()
                    tm[mlp_key + "up_proj.weight"] = tm["language_model"]["encoder"][mlp_key + "proj.weight"][
                                                     mlp_len:].clone()
            merge_weight_config2 = MergeWeightConfig(entire_model, tp_models, 'layers.{}.mlp.gate_proj.weight',
                                                     "language_model.layers.{}.mlp.gate_proj.weight",
                                                     pp_i, g_i, dim=0)
            merge_weight(merge_weight_config2)
            merge_weight_config3 = MergeWeightConfig(entire_model, tp_models, 'layers.{}.mlp.dense_h_to_4h.weight',
                                                     "language_model.layers.{}.mlp.up_proj.weight",
                                                     pp_i, g_i, dim=0)
            merge_weight(merge_weight_config3)
            merge_weight_config4 = MergeWeightConfig(entire_model, tp_models, 'layers.{}.mlp.dense_4h_to_h.weight',
                                                     "language_model.layers.{}.mlp.down_proj.weight",
                                                     pp_i, g_i, dim=1)
            merge_weight(merge_weight_config4)
    return entire_model


def generate_ascendspeed_weights(config):
    tp_size = config.tp_size
    pp_size = config.pp_size
    output_model_dir = config.output_model_dir
    make_vocab_size_divisible_by = config.make_vocab_size_divisible_by
    num_heads = config.num_heads
    num_layer = config.num_layer
    hid_size = config.hid_size
    added_token_num = config.added_token_num

    release_model_dir = os.path.join(output_model_dir, "release")
    check_divisible(num_heads, tp_size)
    check_divisible(num_layer, pp_size)
    pp_n_layer = num_layer // pp_size

    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            model_dic = {"checkpoint_version": 3.0}
            rank_model = {
                "language_model": {"embedding": {"word_embeddings": {"weight": {}}}, "encoder": {}, "output_layer": {}}}

            emb_w = get_weight_from_name("embedding.word_embeddings.weight")
            emb_w = pad_embed(emb_w, make_vocab_size_divisible_by, tp_size, added_token_num)

            if pp_rank == 0:
                rank_model["language_model"]["embedding"]["word_embeddings"]["weight"] = row_split(emb_w, tp_size, tp_rank)

            if pp_rank == pp_size - 1:
                rank_model["language_model"]["encoder"]["final_layernorm.weight"] = get_weight_from_name(
                    "language_model.final_layernorm.weight").clone()
                rank_model["language_model"]["output_layer"]["weight"] = row_split(
                    pad_embed(get_weight_from_name("lm_head.lm_head.weight"), make_vocab_size_divisible_by,
                              tp_size, added_token_num), tp_size, tp_rank)

            for pp_i in range(pp_n_layer):
                g_i = pp_n_layer * pp_rank + pp_i
                qkv_key = f"language_model.layers.{g_i}.attention.query_key_value.weight"
                qw = row_split(get_weight_from_name(qkv_key + "_query"), tp_size, tp_rank)
                kw = row_split(get_weight_from_name(qkv_key + "_key"), tp_size, tp_rank)
                vw = row_split(get_weight_from_name(qkv_key + "_value"), tp_size, tp_rank)
                permute_w = permute_qkv_weight(torch.cat([qw, kw, vw], dim=0), (num_heads, hid_size, tp_size, args.num_kv_heads))
                rank_model["language_model"]["encoder"][f"layers.{pp_i}.self_attention.query_key_value.weight"] = permute_w

                rank_model["language_model"]["encoder"][f"layers.{pp_i}.self_attention.dense.weight"] = column_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.attention.dense.weight"), tp_size, tp_rank)

                gate_proj = row_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.gate_proj.weight"), tp_size, tp_rank)
                up_proj = row_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.up_proj.weight"), tp_size, tp_rank)
                if args.merge_mlp:
                    rank_model["language_model"]["encoder"][f"layers.{pp_i}.mlp.proj.weight"] = torch.cat(
                        [gate_proj, up_proj], 0).contiguous().clone()
                else:
                    rank_model["language_model"]["encoder"][f"layers.{pp_i}.mlp.gate_proj.weight"] = gate_proj
                    rank_model["language_model"]["encoder"][f"layers.{pp_i}.mlp.dense_h_to_4h.weight"] = up_proj

                rank_model["language_model"]["encoder"][f"layers.{pp_i}.mlp.dense_4h_to_h.weight"] = column_split(
                    get_weight_from_name(f"language_model.layers.{g_i}.mlp.down_proj.weight"), tp_size, tp_rank)

                rank_model["language_model"]["encoder"][f"layers.{pp_i}.input_layernorm.weight"] = get_weight_from_name(
                    f"language_model.layers.{g_i}.input_layernorm.weight").clone()
                rank_model["language_model"]["encoder"][f"layers.{pp_i}.post_attention_layernorm.weight"] = get_weight_from_name(
                    f"language_model.layers.{g_i}.post_attention_layernorm.weight").clone()
            if tp_rank == 0 and pp_rank == 0:
                print_model(rank_model)

            save_ascendspeed_model_config = SaveAscendspeedModelConfig(model_dic, rank_model, pp_size,
                                                                       tp_rank, pp_rank, release_model_dir)
            save_ascendspeed_model(save_ascendspeed_model_config)


def print_result(arg):
    logging.info("=" * 100)
    logging.info(
        "weight converted from (tp=%s,pp=%s) to (tp=%s,pp=%s) success.. the converted weights are stored in %s",
        str(arg.src_tensor_model_parallel_size), str(arg.src_pipeline_model_parallel_size),
        str(arg.tgt_tensor_model_parallel_size), str(arg.tgt_pipeline_model_parallel_size),
        str(arg.output_model_dir)
    )
    logging.info("=" * 100)


if __name__ == '__main__':
    if args.type in model_config:
        n_layer, hidden_size, n_heads = model_config[args.type]
    else:
        n_layer, hidden_size, n_heads = args.num_layers, args.hidden_size, args.num_heads

    if args.num_kv_heads is None:
        args.num_kv_heads = n_heads

    check_model_dir(args.input_model_dir, args.src_tensor_model_parallel_size, args.src_pipeline_model_parallel_size)
    make_ascendspeed_model_dirs(args.output_model_dir)

    merge_pp_tp_models_config = MergePpTpMergeConfig(args.src_pipeline_model_parallel_size,
                                                     args.src_tensor_model_parallel_size,
                                                     args.input_model_dir, args.orig_vocab_size,
                                                     n_heads, n_layer, hidden_size)
    _ = merge_pp_tp_models(merge_pp_tp_models_config)

    gen_ascend_weights_config = GenAscendWeightsConfig(args.tgt_tensor_model_parallel_size,
                                                       args.tgt_pipeline_model_parallel_size, args.output_model_dir,
                                                       args.make_vocab_size_divisible_by, n_heads,
                                                       n_layer, hidden_size, args.added_token_num)
    generate_ascendspeed_weights(gen_ascend_weights_config)
    print_result(args)
