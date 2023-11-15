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


import timeit
import math
import unittest
import logging
import torch
import torch_npu
from wrapt_timeout_decorator import timeout
from ascendspeed.core.transformer.module.triangle_attention import TriangleAttention


class TestTriangleAttention(unittest.TestCase):

    @timeout(1200)
    def create_data(self, bsz, hn, seq, hd):
        device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
        query_layer = torch.rand(bsz, hn, seq, hd).half().to(device)
        key_layer = torch.rand(bsz, hn, seq, hd).half().to(device)
        value_layer = torch.rand(bsz, hn, seq, hd).half().to(device)
        mask = torch.triu(torch.ones(1, 1, seq, seq), diagonal=1).to(device)
        mask = mask.bool()
        data_tuple = (query_layer, key_layer, value_layer, mask)
        return data_tuple

    @timeout(1200)
    def test_function(self):
        logging.getLogger().setLevel(logging.INFO)
        logging.info("triangle attention test start")
        bsz = 8
        hn = 8
        seq = 2048
        hd = 1024
        query_layer, key_layer, value_layer, mask = self.create_data(bsz, hn, seq, hd)
        triangle_attn = TriangleAttention(block_size=512, masked_softmax_func=None, dropout_func=None)
        output = triangle_attn(query_layer, key_layer, value_layer, mask)
        logging.info("triangle attention test end")

    @timeout(1200)
    def test_performance(self):
        bsz = 8
        hn = 8
        seq = 2048
        hd = 1024
        query_layer, key_layer, value_layer, mask = self.create_data(bsz, hn, seq, hd)
        iterations = 100
        triangle_attn = TriangleAttention(block_size=512, masked_softmax_func=None, dropout_func=None)
        # warm up
        for _ in range(iterations):
            output = triangle_attn(query_layer, key_layer, value_layer, mask)
        torch.npu.synchronize()
        s_time = timeit.default_timer()
        for _ in range(iterations):
            output = triangle_attn(query_layer, key_layer, value_layer, mask)
        time_elapsed = (timeit.default_timer() - s_time) / iterations


