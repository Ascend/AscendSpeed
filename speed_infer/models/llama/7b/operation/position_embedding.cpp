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
#include "position_embedding.h"

namespace atb_speed {
namespace llama_7b {
enum PositionEmbeddingTensorId {
    INPUT_ID = 0,                          // [batch, seqLen, hiddenSize], half
    IN_POSITION_ID,                        // [batch, seqLen], int64
    IN_COS_TABLE_ID,                       // [1, 1, maxseqLen, headDim], half
    IN_SIN_TABLE_ID,                       // [1, 1, maxseqLen, headDim], half
    OUTPUT_TRANSPOSE_ID,                   // [batch, seqlen, headNum, headDim], half
    INTERMEDIATE_INPUT_TRANSPOSED_ID,      // [batch, headNum, seqLen, headDim], half
    INTERMEDIATE_COS_ID,                   // [batch, seqlen, headDim], half
    INTERMEDIATE_SIN_ID,                   // [batch, seqlen, headDim], half
    INTERMEDIATE_INPUT_TRANSPOSED0_ID,     // [batch, headNum, seqLen, headDim / 2], half
    INTERMEDIATE_INPUT_TRANSPOSED1_ID,     // [batch, headNum, seqLen, headDim / 2], half
    INTERMEDIATE_INPUT_TRANSPOSED1_NEG_ID, // [batch, headNum, seqLen, headDim / 2], half
    INTERMEDIATE_INPUT_ROTATE_ID,          // [batch, headNum, seqLen, headDim], half
    INTERMEDIATE_MUL0_ID,                  // [batch, headNum, seqLen, headDim], half
    INTERMEDIATE_MUL1_ID,                  // [batch, headNum, seqLen, headDim], half
    INTERMEDIATE_INPUT_EMBEDDED_ID         // [batch, headNum, seqLen, headDim], half
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 10;

void Squeeze01(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dims[0] == 1) {
        newShape.dimNum = oldShape.dimNum - 2;
        for (size_t i = 0; i < newShape.dimNum; i++) {
            newShape.dims[i] = oldShape.dims[i + 2];
        }
    } else {
        newShape = oldShape;
    }
}

void unsqueezeCosSinView(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = 1;
    newShape.dims[2] = oldShape.dims[1];
    newShape.dims[3] = oldShape.dims[2];
}

atb::Status PositionEmbedding(const PositionEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &inputTransposedNode = opGraph.nodes.at(nodeId++);
    auto &embeddingCos0Node = opGraph.nodes.at(nodeId++);
    auto &embeddingSin0Node = opGraph.nodes.at(nodeId++);
    auto &splitNode = opGraph.nodes.at(nodeId++);
    auto &negNode = opGraph.nodes.at(nodeId++);
    auto &concat01Node = opGraph.nodes.at(nodeId++);
    auto &mul0Node = opGraph.nodes.at(nodeId++);
    auto &mul1Node = opGraph.nodes.at(nodeId++);
    auto &addNode = opGraph.nodes.at(nodeId++);
    auto &outTransposedNode = opGraph.nodes.at(nodeId++);

    atb::infer::TransposeParam inputTransposeParam;
    inputTransposeParam.perm = {0, 2, 1, 3}; // 对dim1、dim2维度进行转置
    CreateOperation(inputTransposeParam, &inputTransposedNode.operation);
    inputTransposedNode.inTensorIds = {INPUT_ID};
    inputTransposedNode.outTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED_ID};
    inputTransposedNode.inTensorReshapeFuncs.resize(inputTransposedNode.inTensorIds.size());
    inputTransposedNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb::infer::GatherParam embeddingCos0Param;
    CreateOperation(embeddingCos0Param, &embeddingCos0Node.operation);
    embeddingCos0Node.inTensorIds = {IN_COS_TABLE_ID, IN_POSITION_ID};
    embeddingCos0Node.outTensorIds = {INTERMEDIATE_COS_ID};
    embeddingCos0Node.inTensorReshapeFuncs.resize(embeddingCos0Node.inTensorIds.size());
    embeddingCos0Node.inTensorReshapeFuncs.at(0) = &Squeeze01;

    atb::infer::GatherParam embeddingSin0Param;
    CreateOperation(embeddingSin0Param, &embeddingSin0Node.operation);
    embeddingSin0Node.inTensorIds = {IN_SIN_TABLE_ID, IN_POSITION_ID};
    embeddingSin0Node.outTensorIds = {INTERMEDIATE_SIN_ID};
    embeddingSin0Node.inTensorReshapeFuncs.resize(embeddingSin0Node.inTensorIds.size());
    embeddingSin0Node.inTensorReshapeFuncs.at(0) = &Squeeze01;

    atb::infer::SplitParam splitParam;
    splitParam.splitDim = 3; // 3: 在第dim3维上进行切分
    splitParam.splitNum = 2; // 2: 进行二等分
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED_ID};
    splitNode.outTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED0_ID, INTERMEDIATE_INPUT_TRANSPOSED1_ID};

    atb::infer::ElewiseParam negParam;
    negParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    negParam.mulsParam.varAttr = -1;
    CreateOperation(negParam, &negNode.operation);
    negNode.inTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED1_ID};
    negNode.outTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED1_NEG_ID};

    atb::infer::ConcatParam concat01Param;
    concat01Param.concatDim = 3;
    CreateOperation(concat01Param, &concat01Node.operation);
    concat01Node.inTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED1_NEG_ID, INTERMEDIATE_INPUT_TRANSPOSED0_ID};
    concat01Node.outTensorIds = {INTERMEDIATE_INPUT_ROTATE_ID};

    atb::infer::ElewiseParam mul0Param;
    mul0Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mul0Param, &mul0Node.operation);
    mul0Node.inTensorIds = {INTERMEDIATE_INPUT_TRANSPOSED_ID, INTERMEDIATE_COS_ID};
    mul0Node.outTensorIds = {INTERMEDIATE_MUL0_ID};
    mul0Node.inTensorReshapeFuncs.resize(mul0Node.inTensorIds.size());
    mul0Node.inTensorReshapeFuncs.at(1) = &unsqueezeCosSinView;

    atb::infer::ElewiseParam mul1Param;
    mul1Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mul1Param, &mul1Node.operation);
    mul1Node.inTensorIds = {INTERMEDIATE_INPUT_ROTATE_ID, INTERMEDIATE_SIN_ID};
    mul1Node.outTensorIds = {INTERMEDIATE_MUL1_ID};
    mul1Node.inTensorReshapeFuncs.resize(mul1Node.inTensorIds.size());
    mul1Node.inTensorReshapeFuncs.at(1) = &unsqueezeCosSinView;

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &addNode.operation);
    addNode.inTensorIds = {INTERMEDIATE_MUL0_ID, INTERMEDIATE_MUL1_ID};
    addNode.outTensorIds = {INTERMEDIATE_INPUT_EMBEDDED_ID};

    // trans [batch,headNum,seqlen,headDim] to [batch,seqlen, headNum, headDim]
    atb::infer::TransposeParam outTransposedParam;
    outTransposedParam.perm = {0, 2, 1, 3};
    CreateOperation(outTransposedParam, &outTransposedNode.operation);
    outTransposedNode.inTensorIds = {INTERMEDIATE_INPUT_EMBEDDED_ID};
    outTransposedNode.outTensorIds = {OUTPUT_TRANSPOSE_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;
        outTensorDescs.at(0).shape.dims[3] = inTensorDescs.at(0).shape.dims[2] / param.headNum;
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed