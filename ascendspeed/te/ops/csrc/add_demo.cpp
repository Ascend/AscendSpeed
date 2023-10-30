// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

#include "atb/operation_torch.h"
#include "common.h"

using namespace std;
torch::Tensor add(const torch::Tensor &input1, const torch::Tensor &input2)
{
    OperationTorch atb("AddOperation");

    std::vector<torch::Tensor> atInTensors;
    std::vector<torch::Tensor> atOutTensors;
    atInTensors.push_back(input1);
    atInTensors.push_back(input2);
    atb.SetParam("{\"scale\": 1}");
    atOutTensors = atb.Execute(atInTensors);

    return atOutTensors[0];
}
