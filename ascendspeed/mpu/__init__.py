# coding=utf-8
# Copyright (c) 2020, HUAWEI CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from .data import broadcast_data

from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding
from .layers import (set_tensor_model_parallel_attributes,
                     set_defaults_if_not_set_tensor_model_parallel_attributes,
                     copy_tensor_model_parallel_attributes)
                     
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region

from .random import checkpoint
from .random import get_cuda_rng_tracker
from .random import init_checkpointed_activations_memory_buffer
from .random import model_parallel_cuda_manual_seed
from .random import reset_checkpointed_activations_memory_buffer
from .random import gather_split_1d_tensor
from .random import split_tensor_into_1d_equal_chunks

from ascendspeed.core.parallel_state import *

