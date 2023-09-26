# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION. All rights reserved.
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

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
import deepspeed.runtime.utils as ds_utils
from deepspeed.utils import logger
import torch.nn as nn
from ascendspeed import get_args


def bloom_manual_layer_allocation():
    args = get_args()
    parts = [0]
    for nlayer in args.manual_layers:
        parts.append(parts[-1] + nlayer)
    for i in range(1, len(parts)):
        parts[i] += 3

    parts[-1] += 4
    return parts


def use_bloom_manual_layer_allocation():
    args = get_args()
    return args.use_manual_layer_allocation


class ManuallyAllocatedPipelineModule(PipelineModule):
    def _partition_layers(self, method='uniform'):
        super(ManuallyAllocatedPipelineModule, self)._partition_layers(method)
        stage_id = self._topo.get_coord(self.global_rank).pipe
        method = method.lower()
        if method.startswith('type:'):
            if use_bloom_manual_layer_allocation():
                parts = bloom_manual_layer_allocation()
                self._set_bounds(start=parts[stage_id], stop=parts[stage_id + 1])
