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

import unittest
import json
import torch
from ascendspeed.core import parallel_state
from ascendspeed.core.memory.auto_recomputing.autorecompute_solver import solve_graph


class TestAutoRecomputing(unittest.TestCase):
    transformer_layer_info_fmt = """
     {
        "name": "%d",
        "layers": [
        {
            "name": "input_layernorm",
            "memory": 384,
            "time": 1.9710063934326172,
            "input": 64.0,
            "peak_memory": 402705408,
            "forward_cnt": 2,
            "pre_total_time": 3.9420127868652344
        }, {
            "name": "attention",
            "layers": [{
                "name": "query_key_value",
                "memory": 192,
                "time": 9.331226348876953,
                "input": 64.0,
                "peak_memory": 402654208,
                "forward_cnt": 2,
                "pre_total_time": 18.662452697753906
            }, {
                "name": "rotary_emb",
                "memory": 0,
                "time": 1.7354488372802734,
                "input": 64.0,
                "peak_memory": 0,
                "forward_cnt": 2,
                "pre_total_time": 3.470897674560547
            }, {
                "name": "triangle_attn",
                "layers": [{
                    "name": "scaled_masked_softmax",
                    "memory": 512,
                    "time": 465.08251536976206,
                    "input": 516.0,
                    "peak_memory": 542107136,
                    "forward_cnt": 11,
                    "pre_total_time": 5115.907669067383
                }],
                "memory": 1664,
                "time": 22.87912368774414,
                "input": 208.0,
                "peak_memory": 2818581504,
                "forward_cnt": 2,
                "pre_total_time": 45.75824737548828
            }, {
                "name": "dense",
                "memory": 64,
                "time": 8.333802223205566,
                "input": 64.0,
                "peak_memory": 536871936,
                "forward_cnt": 2,
                "pre_total_time": 16.667604446411133
            }],
            "memory": 1792,
            "time": 50.97508430480957,
            "input": 80.0,
            "peak_memory": 2684364288,
            "forward_cnt": 2,
            "pre_total_time": 101.95016860961914
        }, {
            "name": "post_attention_layernorm",
            "memory": 384,
            "time": 1.8906593322753906,
            "input": 64.0,
            "peak_memory": 402705408,
            "forward_cnt": 2,
            "pre_total_time": 3.7813186645507812
        }, {
            "name": "mlp",
            "layers": [{
                "name": "gate_proj",
                "memory": 172,
                "time": 9.36591625213623,
                "input": 64.0,
                "peak_memory": 360711168,
                "forward_cnt": 2,
                "pre_total_time": 18.73183250427246
            }, {
                "name": "up_proj",
                "memory": 172,
                "time": 8.879423141479492,
                "input": 64.0,
                "peak_memory": 360711168,
                "forward_cnt": 2,
                "pre_total_time": 17.758846282958984
            }, {
                "name": "down_proj",
                "memory": 64,
                "time": 13.797521591186523,
                "input": 172.0,
                "peak_memory": 536871936,
                "forward_cnt": 2,
                "pre_total_time": 27.595043182373047
            }],
            "memory": 752,
            "time": 38.39600086212158,
            "input": 64.0,
            "peak_memory": 1258294272,
            "forward_cnt": 2,
            "pre_total_time": 76.79200172424316
        }],
        "memory": 3312,
        "time": 100.17907619476318,
        "input": 64.0,
        "peak_memory": 3942760960,
        "forward_cnt": 2,
        "pre_total_time": 200.35815238952637
    }
    """
    module_all_fmt = """
    {
        "module": [],
        "layers": [{
            "name": "module",
            "layers": [
            {
                "name": "module",
                "layers": [
                {
                    "name": "embedding",
                    "layers": [
                    {
                        "name": "word_embeddings",
                        "memory": 256,
                        "time": 13.043999671936035,
                        "input": 0.25,
                        "peak_memory": 268797952,
                        "forward_cnt": 2,
                        "pre_total_time": 26.08799934387207
                    }],
                    "memory": 64,
                    "time": 16.85166358947754,
                    "input": 0.25,
                    "peak_memory": 604310016,
                    "forward_cnt": 2,
                    "pre_total_time": 33.70332717895508
                }, 
                {
                    "name": "language_model",
                    "layers": [
                    {
                        "name": "layers",
                        "layers": [%s]
                    }],
                    "memory": 4336,
                    "time": 1621.1401224136353,
                    "input": 80.0,
                    "peak_memory": 5331085312,
                    "forward_cnt": 2,
                    "pre_total_time": 3242.2802448272705
                }],
                "memory": 4336,
                "time": 1642.3271894454956,
                "input": 16.25,
                "peak_memory": 5398523392,
                "forward_cnt": 2,
                "pre_total_time": 3284.654378890991
            }],
            "memory": 4336,
            "time": 1645.2174186706543,
            "input": 16.25,
            "peak_memory": 5398523392,
            "forward_cnt": 2,
            "pre_total_time": 3290.4348373413086
        }],
        "used_mem": 16600,
        "max_device_memory": 58960
    }
    """

    def get_module(self, size):
        module_layers = [self.transformer_layer_info_fmt % i for i in range(size)]
        module_layers_context = self.module_all_fmt % (",".join(module_layers))
        module = json.loads(module_layers_context)
        return module

    def get_transformer_layers(self, module):
        transformer_layers = None
        for sub_module in module["layers"]:
            if sub_module["name"] == "layers":
                transformer_layers = sub_module["layers"]
                break
            if "layers" not in sub_module:
                continue
            transformer_layers = self.get_transformer_layers(sub_module)
        return transformer_layers

    @staticmethod
    def is_recompute_module(module):
        if "recompute" in module and module["recompute"]:
            return True
        return False

    def get_module_recompute_layer(self, module):
        recompute_module_layer = []
        for sub_module in module:
            if self.is_recompute_module(sub_module):
                recompute_module_layer.append(sub_module["name"])
        return recompute_module_layer

    def assert_policy(self, module, policy):
        transformer_layers = self.get_transformer_layers(module)
        for module in transformer_layers:
            # n_full
            if self.is_recompute_module(module):
                if "n_full" not in policy:
                    return False
                if policy["n_full"] <= 0:
                    return False
                policy["n_full"] -= 1
                continue
            sub_module_recompute_layer = self.get_module_recompute_layer(module["layers"])
            # n_without
            if len(sub_module_recompute_layer) == 0:
                if "n_without" not in policy:
                    return False
                if policy["n_without"] <= 0:
                    return False
                policy["n_without"] -= 1
                continue
            # n_selective
            if "n_selective" not in policy or "n_selective_recompute_nodes" not in policy:
                return False
            if policy["n_selective"] <= 0:
                return False
            if len(sub_module_recompute_layer) != len(policy["n_selective_recompute_nodes"]):
                return False
            if len(set(sub_module_recompute_layer) | set(policy["n_selective_recompute_nodes"])) != len(
                    policy["n_selective_recompute_nodes"]):
                return False
            policy["n_selective"] -= 1
        return True

    def do_solve_graph(self, layer_num, pp, device_memory):
        module = self.get_module(layer_num)
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        solve_graph(module, pp, device_memory)
        return module

    def test_solve_graph_by_module_10_layer_pp_2_52G(self):
        print("=== start to test solve graph: module 10 layer, pp 2, memory 52GB ===")
        module = self.do_solve_graph(10, 2, 52 * 1024)
        policy = {
            "n_without": 4,
            "n_full": 1,
            "n_selective": 5,
            "n_selective_recompute_nodes": ["input_layernorm", "attention", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_2_54G(self):
        print("=== start to test solve graph: module 10 layer, pp 2, memory 54GB ===")
        module = self.do_solve_graph(10, 2, 54 * 1024)
        policy = {
            "n_without": 1,
            "n_full": 3,
            "n_selective": 6,
            "n_selective_recompute_nodes": ["input_layernorm", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_1_52G(self):
        print("=== start to test solve graph: module 10 layer, pp 1, memory 52GB ===")
        module = self.do_solve_graph(10, 1, 52 * 1024)
        policy = {
            "n_without": 10
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_10_layer_pp_1_54G(self):
        print("=== start to test solve graph: module 10 layer, pp 1, memory 54GB ===")
        module = self.do_solve_graph(10, 1, 54 * 1024)
        policy = {
            "n_without": 10
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_2_52G(self):
        print("=== start to test solve graph: module 32 layer, pp 2, memory 52GB ===")
        module = self.do_solve_graph(32, 2, 52 * 1024)
        policy = {
            "n_full": 13,
            "n_selective": 19,
            "n_selective_recompute_nodes": ["input_layernorm", "attention", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_2_54G(self):
        print("=== start to test solve graph: module 32 layer, pp 2, memory 54GB ===")
        module = self.do_solve_graph(32, 2, 54 * 1024)
        policy = {
            "n_full": 12,
            "n_selective": 20,
            "n_selective_recompute_nodes": ["input_layernorm", "attention", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_1_52G(self):
        print("=== start to test solve graph: module 32 layer, pp 1, memory 52GB ===")
        module = self.do_solve_graph(32, 1, 52 * 1024)
        policy = {
            "n_without": 2,
            "n_selective": 30,
            "n_selective_recompute_nodes": ["input_layernorm", "attention", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))

    def test_solve_graph_by_module_32_layer_pp_1_54G(self):
        print("=== start to test solve graph: module 32 layer, pp 1, memory 54GB ===")
        module = self.do_solve_graph(32, 1, 54 * 1024)
        policy = {
            "n_without": 3,
            "n_selective": 29,
            "n_selective_recompute_nodes": ["input_layernorm", "attention", "post_attention_layernorm"]
        }
        self.assertTrue(self.assert_policy(module, policy))


if __name__ == '__main__':
    unittest.main()
