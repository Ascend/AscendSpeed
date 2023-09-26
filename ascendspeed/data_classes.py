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


from dataclasses import dataclass


@dataclass
class MergeWeightConfig:
    entire_model_dic: dict
    tp_models: list
    k: int
    pp_i: int
    tot_i: int
    dim: int


@dataclass
class MergePpTpMergeConfig:
    pp_size: int
    tp_size: int
    input_model_dir: int
    orig_vocab_size: int
    num_heads: int
    num_layer: int
    hid_size: int


@dataclass
class GenAscendWeightsConfig:
    tp_size: int
    pp_size: int
    output_model_dir: str
    make_vocab_size_divisible_by: int
    num_heads: int
    num_layer: int
    hid_size: int
    added_token_num: int


@dataclass
class GenAscendWeightsAgaConfig:
    tp_size: int
    pp_size: int
    model_type: str
    out_model_dir: str
    make_vocab_size_divisible_by: int
    added_token_num: int


@dataclass()
class SaveAscendspeedModelConfig:
    model_dic: dict
    rank_model: dict
    pp_size: int
    tp_rank: int
    pp_rank: int
    release_model_dir: str
