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

"""Convert weight from huggingface to ascendspeed"""

import argparse
import json
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

from ascendspeed.error_utils import check_divisible
from ascendspeed.data_classes import GenAscendWeightsAgaConfig, SaveAscendspeedModelConfig
logging.basicConfig(level=logging.NOTSET)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir", help="llama native model dir")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir", help="ascendspeed model dir")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128,
                        help="should be consistent with ascendspeed")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--added-token-num", type=int, default=0, help="the number of added tokens")
    parser.add_argument("--type", type=str, choices=["7B", "13B", "30B", "65B"], default="7B")
    parser.add_argument("--pse", type=bool, default=False)
    return parser.parse_args()


model_config = {
    "7B": [32, 4096, 32],
    "13B": [40, 5120, 40],
    "30B": [60, 6656, 52],
    "65B": [80, 8192, 64]
}


args = get_args()
file = os.listdir(args.input_model_dir)
model_files = [f for f in file if f[-4:] == ".bin"]
input_models = {f: torch.load(os.path.join(args.input_model_dir, f), map_location="cpu") for f in model_files}

with open(os.path.join(args.input_model_dir, "pytorch_model.bin.index.json")) as f:
    model_index = json.load(f)
    weight_map = model_index["weight_map"]


def get_weight_from_name(layer_name):
    return input_models[weight_map[layer_name]][layer_name]


def generate_ascendspeed_weights_again(config):
    tp_size = config.tp_size
    pp_size = config.pp_size
    model_type = config.model_type
    output_model_dir = config.out_model_dir
    make_vocab_size_divisible_by = config.make_vocab_size_divisible_by
    added_token_num = config.added_token_num

    if model_type in model_config:
        n_layer, hidden_size, n_heads = model_config[model_type]
    else:
        raise KeyError(f"{model_type} is not in {model_config}")

    check_divisible(n_heads, tp_size)
    check_divisible(n_layer, pp_size)
    pp_n_layer = n_layer // pp_size

    release_model_dir = os.path.join(output_model_dir, "release")

    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            model_dic = {"checkpoint_version": 3.0}
            rank_model = {}
            emb_w = get_weight_from_name("model.embed_tokens.weight")
            emb_w = pad_embed(emb_w, make_vocab_size_divisible_by, tp_size, added_token_num)

            if pp_rank == 0:
                rank_model["embedding.word_embeddings.weight"] = row_split(emb_w, tp_size, tp_rank)

            if pp_rank == pp_size - 1:
                rank_model["language_model.final_layernorm.weight"] = get_weight_from_name("model.norm.weight").clone()
                rank_model["lm_head.lm_head.weight"] = row_split(
                    pad_embed(get_weight_from_name("lm_head.weight"), make_vocab_size_divisible_by,
                              tp_size, added_token_num), tp_size, tp_rank)

            for pp_i in range(pp_n_layer):
                ori_i = pp_n_layer * pp_rank + pp_i
                if args.pse:
                    w_pack = get_weight_from_name(f"model.layers.{ori_i}.self_attn.W_pack.weight")
                    ws = torch.split(w_pack, w_pack.shape[0] // 3)
                    qw = row_split(ws[0], tp_size, tp_rank)
                    kw = row_split(ws[1], tp_size, tp_rank)
                    vw = row_split(ws[2], tp_size, tp_rank)
                else:
                    rank_model[f"language_model.layers.{pp_i}.attention.rotary_emb.inv_freq"] = get_weight_from_name(
                        f"model.layers.{ori_i}.self_attn.rotary_emb.inv_freq")
                    qw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.q_proj.weight"), tp_size, tp_rank)
                    kw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.k_proj.weight"), tp_size, tp_rank)
                    vw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.v_proj.weight"), tp_size, tp_rank)

   
                permute_w = permute_qkv_weight(torch.cat([qw, kw, vw], dim=0), n_heads, hidden_size, tp_size)
                rank_model[f"language_model.layers.{pp_i}.attention.query_key_value.weight"] = permute_w

                rank_model[f"language_model.layers.{pp_i}.attention.dense.weight"] = column_split(
                    get_weight_from_name(f"model.layers.{ori_i}.self_attn.o_proj.weight"), tp_size, tp_rank)

                rank_model[f"language_model.layers.{pp_i}.mlp.gate_proj.weight"] = row_split(
                    get_weight_from_name(f"model.layers.{ori_i}.mlp.gate_proj.weight"), tp_size, tp_rank)
                rank_model[f"language_model.layers.{pp_i}.mlp.up_proj.weight"] = row_split(
                    get_weight_from_name(f"model.layers.{ori_i}.mlp.up_proj.weight"), tp_size, tp_rank)
                rank_model[f"language_model.layers.{pp_i}.mlp.down_proj.weight"] = column_split(
                    get_weight_from_name(f"model.layers.{ori_i}.mlp.down_proj.weight"), tp_size, tp_rank)

                rank_model[f"language_model.layers.{pp_i}.input_layernorm.weight"] = get_weight_from_name(
                    f"model.layers.{ori_i}.input_layernorm.weight").clone()
                rank_model[f"language_model.layers.{pp_i}.post_attention_layernorm.weight"] = get_weight_from_name(
                    f"model.layers.{ori_i}.post_attention_layernorm.weight").clone()
            if tp_rank == 0 and pp_rank == 0:
                print_model(rank_model)

            save_ascendspeed_model_config = SaveAscendspeedModelConfig(model_dic, rank_model, pp_size,
                                                                       tp_rank, pp_rank, release_model_dir)
            save_ascendspeed_model(save_ascendspeed_model_config)
    logging.info("=" * 100)
    logging.info("llama %s weight has been converted to ascendspeed format"
                 " which stored in %s", str(model_type), str(output_model_dir))
    logging.info("=" * 100)


if __name__ == '__main__':
    make_ascendspeed_model_dirs(args.output_model_dir)
    generate_ascendspeed_weights_again_config = GenAscendWeightsAgaConfig(args.tensor_model_parallel_size,
                                                                          args.pipeline_model_parallel_size, args.type,
                                                                          args.output_model_dir,
                                                                          args.make_vocab_size_divisible_by,
                                                                          args.added_token_num)
    generate_ascendspeed_weights_again(generate_ascendspeed_weights_again_config)
