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

#include "mlp_gate.h"

namespace atb_speed {
namespace common {

template <class T> atb::Status MlpGateLayerBase(const MlpGateParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);

    size_t nodeId = 0;

    auto &matmulUpNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam matmulUpParam = {false, param.transposeB, param.isBias};
    atb::CreateOperation(matmulUpParam, &matmulUpNode.operation);
    if (param.isBias) {
        matmulUpNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID, config.IN_BIAS_UP_ID};
    } else {
        matmulUpNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID};
    }
    matmulUpNode.outTensorIds = {config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID};

    if (param.isPack) {
        auto &splitNode = opGraph.nodes.at(nodeId++);
        atb::infer::SplitParam splitParam;
        splitParam.splitDim = -1; // 2: split最后一维
        splitParam.splitNum = 2;  // 2: 进行二等分
        atb::CreateOperation(splitParam, &splitNode.operation);
        splitNode.inTensorIds = {config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID};
        splitNode.outTensorIds = {config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, config.INTERMEDIATE_SPLIT_OUT_ND_ID};
    } else {
        auto &matmulGateNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam matmulGateParam = {false, param.transposeB, param.isBias};
        atb::CreateOperation(matmulGateParam, &matmulGateNode.operation);
        if (param.isBias) {
            matmulGateNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_GATE_ID, config.IN_BIAS_GATE_ID};
        } else {
            matmulGateNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_GATE_ID};
        }
        matmulGateNode.outTensorIds = {config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID};
    }

    auto &actNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam actParam;
    actParam.activationType = param.activationType;
    atb::CreateOperation(actParam, &actNode.operation);
    actNode.inTensorIds = {config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID};
    actNode.outTensorIds = {config.INTERMEDIATE_ACTIVATION_OUT_ID};

    auto &mulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    if (param.isPack) {
        mulNode.inTensorIds = {config.INTERMEDIATE_ACTIVATION_OUT_ID, config.INTERMEDIATE_SPLIT_OUT_ND_ID};
    } else {
        mulNode.inTensorIds = {config.INTERMEDIATE_ACTIVATION_OUT_ID, config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID};
    }
    mulNode.outTensorIds = {config.INTERMEDIATE_MUL_OUT_ID};

    auto &matmulDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ParallelParam linearParallelParam = {param.rank, param.rankSize,  0, nullptr, param.isBias,
                                                            false,      param.transposeB, param.backend};

    atb_speed::common::RowParallelLinear(linearParallelParam, &matmulDownNode.operation);
    if (param.isBias) {
        matmulDownNode.inTensorIds = {config.INTERMEDIATE_MUL_OUT_ID, config.IN_WEIGHT_DOWN_ID, config.IN_BIAS_DOWN_ID};
    } else {
        matmulDownNode.inTensorIds = {config.INTERMEDIATE_MUL_OUT_ID, config.IN_WEIGHT_DOWN_ID};
    }
    matmulDownNode.outTensorIds = {config.OUT_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status MlpGateLayer(const MlpGateParam &param, atb::Operation **operation)
{
    if (param.isBias && param.isPack) {
        return MlpGateLayerBase(param, operation, MlpGateWithPackAndBias(5, 1, 5, 5)); // 5:in 1:out 5:inter 5:node
    } else if (param.isBias) {
        return MlpGateLayerBase(param, operation, MlpGateWithBias(7, 1, 4, 5)); // 7:in 1:out 4:inter 5:node
    } else if (param.isPack) {
        return MlpGateLayerBase(param, operation, MlpGateWithPack(3, 1, 5, 5)); // 3:in 1:out 5:inter 5:node
    } else {
        return MlpGateLayerBase(param, operation, MlpGate(4, 1, 4, 5)); // 4:in 1:out 4:inter 5:node
    }
}

} // namespace common
} // namespace atb_speed
