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

#include "parallel_layer_v2.h"
#include "mlp_gate_v2.h"

namespace atb_speed {
namespace common {

enum InTensorId : int {
    IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
    IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
    IN_WEIGHT_GATE_ID,                  // [hiddenSize, ffnHiddenSize], half
    IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
    IN_DEQSCALE_UP,                     // quant scale up
    IN_DEQSCALE_GATE,                   // quant scale gete
    IN_DEQSCALE_DOWN,                   // quant scale down
    IN_BIAS_UP_ID,
    IN_BIAS_GATE_ID,
    IN_BIAS_DOWN_ID,
    IN_INDEX_UP,
    IN_INDEX_GATE,
    IN_INDEX_DOWN,
    OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
    INTERMEDIATE_MATMUL_OUT_ID, // [batch, seqLen, ffnHiddenSize], half
    INTERMEDIATE_MATMUL_UP_OUT_ID,   // [batch, seqLen, ffnHiddenSize], half
    INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
    INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, ffnHiddenSize], half
    INTERMEDIATE_SPLIT_OUT_ID,
};

atb::Status MlpGateLayerV2(const MlpGateParamV2 &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = 13;
    opGraph.outTensorNum = 1;
    size_t interTensorNum = 0;
    if (param.isPack) {
        interTensorNum = 5;
    } else {
        interTensorNum = 4;
    }
    opGraph.internalTensorNum = interTensorNum;

    size_t nodeCount = 5;
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;

    auto &matmulUpNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ParallelParamV2 linearUpParam = {param.isBias, false, param.transposeB, param.isQuant};
    linearUpParam.quantParam = param.quantUpParam;
    atb_speed::common::RowParallelLinearV2(linearUpParam, &matmulUpNode.operation);
    matmulUpNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_UP_ID, IN_BIAS_UP_ID, IN_DEQSCALE_UP, IN_INDEX_UP};
    matmulUpNode.outTensorIds = {INTERMEDIATE_MATMUL_UP_OUT_ID};

    if (param.isPack) {
        auto &splitNode = opGraph.nodes.at(nodeId++);
        atb::infer::SplitParam splitParam;
        splitParam.splitDim = -1; // 2: [bs, seq, 2*hidden_size]
        splitParam.splitNum = 2; // 2: 进行二等分
        atb::CreateOperation(splitParam, &splitNode.operation);
        splitNode.inTensorIds = {INTERMEDIATE_MATMUL_UP_OUT_ID};
        splitNode.outTensorIds = {INTERMEDIATE_MATMUL_OUT_ID, INTERMEDIATE_SPLIT_OUT_ID};
    } else {
        auto &matmulGateNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::ParallelParamV2 linearGateParam = {param.isBias, false, param.transposeB, param.isQuant};
        linearGateParam.quantParam = param.quantGateParam;
        atb_speed::common::RowParallelLinearV2(linearGateParam, &matmulGateNode.operation);
        matmulGateNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_GATE_ID,
                                      IN_BIAS_GATE_ID, IN_DEQSCALE_GATE, IN_INDEX_GATE};
        matmulGateNode.outTensorIds = {INTERMEDIATE_MATMUL_OUT_ID};
    }
    auto &actNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam actParam;
    actParam.activationType = param.activationType;
    atb::CreateOperation(actParam, &actNode.operation);
    actNode.inTensorIds = {INTERMEDIATE_MATMUL_OUT_ID};
    actNode.outTensorIds = {INTERMEDIATE_ACTIVATION_OUT_ID};

    auto &mulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mulParam, &mulNode.operation);
    if (param.isPack) {
        mulNode.inTensorIds = {INTERMEDIATE_ACTIVATION_OUT_ID, INTERMEDIATE_SPLIT_OUT_ID};
    } else {
        mulNode.inTensorIds = {INTERMEDIATE_ACTIVATION_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ID};
    }
    mulNode.outTensorIds = {INTERMEDIATE_MUL_OUT_ID};

    auto &matmulDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ParallelParamV2 linearDownParam = {param.isBias, false, param.transposeB, param.isQuant};
    linearDownParam.commParam = param.commDownParam;
    linearDownParam.quantParam = param.quantDownParam;
    linearDownParam.quantParam.isQuantOp = true;
    atb_speed::common::RowParallelLinearV2(linearDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = {INTERMEDIATE_MUL_OUT_ID, IN_WEIGHT_DOWN_ID,
                                  IN_BIAS_DOWN_ID, IN_DEQSCALE_DOWN, IN_INDEX_DOWN};
    matmulDownNode.outTensorIds = {OUT_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);

        outTensorDescs.at(0).dtype = ACL_FLOAT16;
        outTensorDescs.at(0).shape.dimNum = 3; // 3 dim
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // dims 0
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // dims 1
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2]; // dims 2
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed
