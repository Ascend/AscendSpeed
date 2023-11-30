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

import argparse
import json
import os
import sys
import logging
from collections import namedtuple
import torch
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.ckpt_convert.llama.ckpt_utils import (column_split, row_split, pad_embed, permute_qkv_weight,
                                                 print_model, make_ascendspeed_model_dirs, save_ascendspeed_model)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-model-dir", type=str, default="./input_model_dir", help="huggingface model dir")
    parser.add_argument("--output-model-dir", type=str, default="./output_model_dir", help="ascendspeed model dir")
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128,
                        help="should be consistent with ascendspeed")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1, help="degree of tensor model parallel")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="degree of pipeline model parallel")
    parser.add_argument("--type", type=str, choices=["7B", "176B"], default="7B")
    parser.add_argument("--deepspeed", action="store_true", default=True)
    parser.add_argument("--partition-layers", type=str, help="the partition method of model when pipeline is used")
    return parser.parse_args()


model_config = {
    "7B": [30, 4096, 32], # num_layers, hidden_size, num_attention_heads
    "176B": [70, 14336, 112]
}


args = get_args()
files = os.listdir(args.input_model_dir)
model_files = [f for f in files if f[-4:] == '.bin']
input_models = {f: torch.load(os.path.join(args.input_model_dir, f), map_location="cpu") for f in model_files}


with open(os.path.join(args.input_model_dir, "pytorch_model.bin.index.json")) as f:
    model_index = json.load(f)
    weight_map = model_index['weight_map']


def get_weight_from_name(layer_name):
    return input_models[weight_map[layer_name]][layer_name]


def get_partition_layers(model_type, num_layers, pp_size, partition_layers=None):
    if model_type == "7B":
        return [num_layers // pp_size]
    else:
        return list(map(int, partition_layers.split(',')))


def generate_ascendspeed_weights(parallelism_config, make_vocab_size_divisible_by, output_model_dir):
    tp_size, pp_size, partition_layers, model_type = parallelism_config.tp_size, parallelism_config.pp_size,\
        parallelism_config.partition_layers, parallelism_config.model_type

    try:
        num_layers, _, _ = model_config[model_type]
    except KeyError:
        logger.error("model_type error")
        return

    pp_layers = get_partition_layers(model_type, num_layers, pp_size, partition_layers)
    if not pp_layers:
        logger.error("pp_layers is empty")
        return
    logger.info("pp_layers = %s", str(pp_layers))

    release_model_dir = os.path.join(output_model_dir, "release")

    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            model_dic = {'checkpoint_version': 3.0}
            rank_model = {}

            if pp_rank == 0:
                emb_w = get_weight_from_name("word_embeddings.weight")
                emb_w = pad_embed(emb_w, make_vocab_size_divisible_by, tp_size)
                rank_model['tied_modules.embed.word_embeddings.weight'] = row_split(emb_w, tp_size, tp_rank)

                emb_layernorm_w = get_weight_from_name("word_embeddings_layernorm.weight")
                emb_layernorm_b = get_weight_from_name("word_embeddings_layernorm.bias")
                rank_model['tied_modules.embed.word_embeddings.norm.weight'] = emb_layernorm_w
                rank_model['tied_modules.embed.word_embeddings.norm.bias'] = emb_layernorm_b

            if pp_rank == pp_size - 1:
                layer_id = 3 + num_layers + 1 + 1 - 1
                rank_model['{}.final_layernorm.weight'.format(layer_id)] = get_weight_from_name('ln_f.weight').clone()
                rank_model['{}.final_layernorm.bias'.format(layer_id)] = get_weight_from_name('ln_f.bias').clone()

            for i in range(pp_layers[pp_rank]):
                layer_id = sum(pp_layers[:pp_rank]) + i
                ascendspeed_layer_id = sum(pp_layers[:pp_rank]) + i + 3

                rank_model[f"{ascendspeed_layer_id}.input_layernorm.weight"] = \
                    get_weight_from_name(f"h.{layer_id}.input_layernorm.weight").clone()
                rank_model[f"{ascendspeed_layer_id}.input_layernorm.bias"] = \
                    get_weight_from_name(f"h.{layer_id}.input_layernorm.bias").clone()

                rank_model[f"{ascendspeed_layer_id}.self_attention.query_key_value.weight"] = row_split(
                    get_weight_from_name(f"h.{layer_id}.self_attention.query_key_value.weight"), tp_size, tp_rank)
                rank_model[f"{ascendspeed_layer_id}.self_attention.query_key_value.bias"] = row_split(
                    get_weight_from_name(f"h.{layer_id}.self_attention.query_key_value.bias"), tp_size, tp_rank)
                
                rank_model[f"{ascendspeed_layer_id}.self_attention.dense.weight"] = column_split(
                    get_weight_from_name(f"h.{layer_id}.self_attention.dense.weight"), tp_size, tp_rank)
                rank_model[f"{ascendspeed_layer_id}.self_attention.dense.bias"] = get_weight_from_name(
                    f"h.{layer_id}.self_attention.dense.bias").clone()

                rank_model[f"{ascendspeed_layer_id}.post_attention_layernorm.weight"] = get_weight_from_name(
                    f"h.{layer_id}.post_attention_layernorm.weight").clone()
                rank_model[f"{ascendspeed_layer_id}.post_attention_layernorm.bias"] = get_weight_from_name(
                    f"h.{layer_id}.post_attention_layernorm.bias").clone()

                rank_model[f"{ascendspeed_layer_id}.mlp.dense_h_to_4h.weight"] = row_split(get_weight_from_name(
                    f"h.{layer_id}.mlp.dense_h_to_4h.weight"), tp_size, tp_rank)
                rank_model[f"{ascendspeed_layer_id}.mlp.dense_h_to_4h.bias"] = row_split(get_weight_from_name(
                    f"h.{layer_id}.mlp.dense_h_to_4h.bias"), tp_size, tp_rank)

                rank_model[f"{ascendspeed_layer_id}.mlp.dense_4h_to_h.weight"] = column_split(get_weight_from_name(
                    f"h.{layer_id}.mlp.dense_4h_to_h.weight"), tp_size, tp_rank)
                rank_model[f"{ascendspeed_layer_id}.mlp.dense_4h_to_h.bias"] = get_weight_from_name(
                    f"h.{layer_id}.mlp.dense_4h_to_h.bias").clone()

            if tp_rank == 0 and pp_rank == 0:
                print_model(rank_model)

            if not args.deepspeed:
                save_ascendspeed_model(model_dic, rank_model, pp_size, tp_rank, pp_rank, release_model_dir)
            else:
                module_key = 'module'
                model_dic[module_key] = {}
                model_dic[module_key][module_key] = rank_model
                model_dic['dp_world_size'] = 0
                model_dir = release_model_dir
                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir,
                                         f"{'mp_rank_{:02d}'.format(pp_rank * tp_size + tp_rank)}_model_states.pt")
                torch.save(model_dic, ckpt_path)
                logger.info("save %s finished", ckpt_path)


if __name__ == '__main__':
    if args.deepspeed:
        make_ascendspeed_model_dirs(args.output_model_dir, filename="latest")
    else:
        make_ascendspeed_model_dirs(args.output_model_dir)

    ParallelConfig = namedtuple('ParallelConfig', ['tp_size', 'pp_size', 'partition_layers', 'model_type'])
    parallel_config = ParallelConfig(tp_size=args.tensor_model_parallel_size,
                                     pp_size=args.pipeline_model_parallel_size,
                                     partition_layers=args.partition_layers,
                                     model_type=args.type)
    generate_ascendspeed_weights(parallel_config, args.make_vocab_size_divisible_by, args.output_model_dir)
