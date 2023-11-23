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
#include "common_mlp.h"

namespace atb_speed {
namespace llama_7b {
enum CommonMlpTensorId {
    IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
    IN_WEIGHT_GATE_ID,                  // [11008, hiddenSize], half
    IN_WEIGHT_DOWN_ID,                  // [hiddenSize, 11008], half
    IN_WEIGHT_UP_ID,                    // [11008, hiddenSize], half
    OUT_TRANSPOSED_RESULT_ID,           // [batch, seqLen, hiddenSize], half
    INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_SWISH_OUT_ID,          // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, 11008], half
    INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, 11008], half
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t NODE_COUNT = 5;

atb::Status CommonMlp(const CommonMlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &matmulGateNode = opGraph.nodes.at(nodeId++);
    auto &swishNode = opGraph.nodes.at(nodeId++);
    auto &matmulUpNode = opGraph.nodes.at(nodeId++);
    auto &mulNode = opGraph.nodes.at(nodeId++);
    auto &matmulDownNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulGateParam = {false, param.transpose, false};
    CreateOperation(matmulGateParam, &matmulGateNode.operation);
    matmulGateNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_GATE_ID};
    matmulGateNode.outTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ND_ID};

    atb::infer::ActivationParam swishParam;
    swishParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(swishParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ND_ID};
    swishNode.outTensorIds = {INTERMEDIATE_SWISH_OUT_ID};

    atb::infer::LinearParam matmulUpParam = {false, param.transpose, false};
    CreateOperation(matmulUpParam, &matmulUpNode.operation);
    matmulUpNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_UP_ID};
    matmulUpNode.outTensorIds = {INTERMEDIATE_MATMUL_UP_OUT_ND_ID};

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMEDIATE_SWISH_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ND_ID};
    mulNode.outTensorIds = {INTERMEDIATE_MUL_OUT_ID};

    atb::infer::LinearParam matmulDownParam = {false, param.transpose, false};
    CreateOperation(matmulDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = {INTERMEDIATE_MUL_OUT_ID, IN_WEIGHT_DOWN_ID};
    matmulDownNode.outTensorIds = {OUT_TRANSPOSED_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2];
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed