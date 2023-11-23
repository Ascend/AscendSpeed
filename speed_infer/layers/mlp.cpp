/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <atb/atb_infer.h>
#include "atb_speed/log.h"

#include "parallel_layer.h"
#include "mlp.h"

namespace atb_speed {
namespace common {

template <class T>
atb::Status MlpLayerBase(const MlpParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);

    size_t nodeId = 0;

    auto &matmulUpNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam matmulUpParam = { false, param.transpose, param.isBias };
    atb::CreateOperation(matmulUpParam, &matmulUpNode.operation);
    if (param.isBias) {
        matmulUpNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID, config.IN_BIAS_UP_ID };
    } else {
        matmulUpNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID };
    }
    matmulUpNode.outTensorIds = { config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID };

    auto &actNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam actParam;
    actParam.activationType = param.activationType;
    atb::CreateOperation(actParam, &actNode.operation);
    actNode.inTensorIds = { config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID };
    actNode.outTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID };

    auto &matmulDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ParallelParam linearParallelParam = { param.rank, param.rankSize, 0, nullptr,
                                                             param.isBias, false, param.transpose };

    atb_speed::common::RowParallelLinear(linearParallelParam, &matmulDownNode.operation);
    if (param.isBias) {
        matmulDownNode.inTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID,
                                     config.IN_WEIGHT_DOWN_ID, config.IN_BIAS_DOWN_ID };
    } else {
        matmulDownNode.inTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID, config.IN_WEIGHT_DOWN_ID };
    }
    matmulDownNode.outTensorIds = { config.OUT_RESULT_ID };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3; // 3 dim
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // dims 0
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // dims 1
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2]; // dims 2
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status MlpLayer(const MlpParam &param, atb::Operation **operation)
{
    if (param.isBias) {
        return MlpLayerBase(param, operation,
            MlpWithBias(5, 1, 2, 3)); // 5:in 1:out 2:inter 3:node
    } else {
        return MlpLayerBase(param, operation,
            MlpWithoutBias(3, 1, 2, 3)); // 3:in 1:out 2:inter 3:node
    }
}
} // namespace common
} // namespace atb_speed
