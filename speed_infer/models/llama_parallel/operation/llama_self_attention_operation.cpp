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
#include "llama_self_attention_operation.h"
#include <numeric>
#include <cmath>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/log.h"

namespace atb_speed {

enum LlamaSelfAttentionTensorId {
    IN_MIXEDQUERYTENSOR = 0,
    IN_MIXEDKEYTENSOR,
    IN_MIXEDVALUETENSOR,
    IN_ATTENTIONMASKTENSOR,
    OUT_CONTEXTOUTTENSOR,
    OUT_PRESENTKEYTENSOR,
    OUT_PRESENTVALUETENSOR,
    INTERMIDATE_PERMUTEDQ,
    INTERMIDATE_PERMUTEDK,
    INTERMIDATE_PERMUTEDV,
    INTERMIDATE_TRANSPOSEDK,
    INTERMIDATE_BMMQKOUT,
    INTERMIDATE_MULSOUT,
    INTERMIDATE_ATTENTIONSCORES,
    INTERMIDATE_ATTENTIONPROBS,
    INTERMIDATE_BMMVOUT,
    INTERMIDATE_CONTEXTOUT,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 13;
static uint64_t DIM3 = 3;

atb::Status CreateLlamaSelfAttentionOperation(const llamaSelfAttentionParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << "CreateLlamaSelfAttentionOperation called";

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &permuteQNode = opGraph.nodes.at(nodeId++);
    atb::Node &permuteKNode = opGraph.nodes.at(nodeId++);
    atb::Node &permuteVNode = opGraph.nodes.at(nodeId++);
    atb::Node &permutePresentKNode = opGraph.nodes.at(nodeId++);
    atb::Node &permutePresentVNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposePresentKNode = opGraph.nodes.at(nodeId++);
    atb::Node &bmmQKNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulsNode = opGraph.nodes.at(nodeId++);
    atb::Node &addMaskNode = opGraph.nodes.at(nodeId++);
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::Node &bmmVNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeContext1Node = opGraph.nodes.at(nodeId++);
    atb::Node &transposeContext2Node = opGraph.nodes.at(nodeId++);

    atb::infer::TransposeParam permuteQNodeParam = {{1, 2, 0, 3}};
    CreateOperation(permuteQNodeParam, &permuteQNode.operation);
    permuteQNode.inTensorIds = {IN_MIXEDQUERYTENSOR};
    permuteQNode.outTensorIds = {INTERMIDATE_PERMUTEDQ};

    atb::infer::TransposeParam permuteKNodeParam = {{1, 2, 0, 3}};
    CreateOperation(permuteKNodeParam, &permuteKNode.operation);
    permuteKNode.inTensorIds = {IN_MIXEDKEYTENSOR};
    permuteKNode.outTensorIds = {INTERMIDATE_PERMUTEDK};

    atb::infer::TransposeParam permuteVNodeParam = {{1, 2, 0, 3}};
    CreateOperation(permuteVNodeParam, &permuteVNode.operation);
    permuteVNode.inTensorIds = {IN_MIXEDVALUETENSOR};
    permuteVNode.outTensorIds = {INTERMIDATE_PERMUTEDV};

    atb::infer::TransposeParam permutePresentKNodeParam = {{2, 0, 1, 3}};
    CreateOperation(permutePresentKNodeParam, &permutePresentKNode.operation);
    permutePresentKNode.inTensorIds = {INTERMIDATE_PERMUTEDK};
    permutePresentKNode.outTensorIds = {OUT_PRESENTKEYTENSOR};

    atb::infer::TransposeParam permutePresentVNodeParam = {{2, 0, 1, 3}};
    CreateOperation(permutePresentVNodeParam, &permutePresentVNode.operation);
    permutePresentVNode.inTensorIds = {INTERMIDATE_PERMUTEDV};
    permutePresentVNode.outTensorIds = {OUT_PRESENTVALUETENSOR};

    atb::infer::TransposeParam transposePresentKNodeParam = {{0, 1, 3, 2}};
    CreateOperation(transposePresentKNodeParam, &transposePresentKNode.operation);
    transposePresentKNode.inTensorIds = {INTERMIDATE_PERMUTEDK};
    transposePresentKNode.outTensorIds = {INTERMIDATE_TRANSPOSEDK};

    atb::infer::MatmulParam bmmQKNodeParam = {false, !param.transpose};
    CreateOperation(bmmQKNodeParam, &bmmQKNode.operation);
    bmmQKNode.inTensorIds = {INTERMIDATE_PERMUTEDQ, INTERMIDATE_TRANSPOSEDK};
    bmmQKNode.outTensorIds = {INTERMIDATE_BMMQKOUT};
    bmmQKNode.inTensorReshapeFuncs.resize(bmmQKNode.inTensorIds.size());
    bmmQKNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };
    bmmQKNode.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };

    atb::infer::ElewiseParam mulsNodeParam;
    mulsNodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    mulsNodeParam.mulsParam.varAttr = 1.0 / sqrt(param.dk);
    CreateOperation(mulsNodeParam, &mulsNode.operation);
    mulsNode.inTensorIds = {INTERMIDATE_BMMQKOUT};
    mulsNode.outTensorIds = {INTERMIDATE_MULSOUT};

    atb::infer::ElewiseParam addMaskNodeParam;
    addMaskNodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addMaskNodeParam, &addMaskNode.operation);
    addMaskNode.inTensorIds = {IN_ATTENTIONMASKTENSOR, INTERMIDATE_MULSOUT};
    addMaskNode.outTensorIds = {INTERMIDATE_ATTENTIONSCORES};
    addMaskNode.inTensorReshapeFuncs.resize(addMaskNode.inTensorIds.size());
    addMaskNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };

    atb::infer::SoftmaxParam softMaxNodeParam;
    softMaxNodeParam.axes = {-1};
    CreateOperation(softMaxNodeParam, &softMaxNode.operation);
    softMaxNode.inTensorIds = {INTERMIDATE_ATTENTIONSCORES};
    softMaxNode.outTensorIds = {INTERMIDATE_ATTENTIONPROBS};

    atb::infer::MatmulParam bmmVNodeParam = {false, !param.transpose};
    CreateOperation(bmmVNodeParam, &bmmVNode.operation);
    bmmVNode.inTensorIds = {INTERMIDATE_ATTENTIONPROBS, INTERMIDATE_PERMUTEDV};
    bmmVNode.outTensorIds = {INTERMIDATE_BMMVOUT};
    bmmVNode.inTensorReshapeFuncs.resize(bmmVNode.inTensorIds.size());
    bmmVNode.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };

    atb::infer::TransposeParam transposeContext1NodeParam = {{0, 2, 1, 3}};
    CreateOperation(transposeContext1NodeParam, &transposeContext1Node.operation);
    transposeContext1Node.inTensorIds = {INTERMIDATE_BMMVOUT};
    transposeContext1Node.outTensorIds = {INTERMIDATE_CONTEXTOUT};
    transposeContext1Node.inTensorReshapeFuncs.resize(transposeContext1Node.inTensorIds.size());
    transposeContext1Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] / param.headNum;
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    atb::infer::TransposeParam transposeContext2NodeParam = {{1, 0, 2}};
    CreateOperation(transposeContext2NodeParam, &transposeContext2Node.operation);
    transposeContext2Node.inTensorIds = {INTERMIDATE_CONTEXTOUT};
    transposeContext2Node.outTensorIds = {OUT_CONTEXTOUTTENSOR};
    transposeContext2Node.inTensorReshapeFuncs.resize(transposeContext2Node.inTensorIds.size());
    transposeContext2Node.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = oldShape.dims[2] * oldShape.dims[3];
    };

    ATB_LOG(INFO) << "CreateLlamaSelfAttentionOperation Operate End";

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs[0];
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs[0].shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs[0].shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs[0].shape.dims[2] * inTensorDescs.at(0).shape.dims[DIM3];

        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(2) = inTensorDescs.at(2);

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed