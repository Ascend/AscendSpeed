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
#include "position_embedding_fusion.h"

namespace atb_speed {
namespace llama_7b {
enum PositionEmbedding1DSplitTensorId {
    MIXEDQKV = 0,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_SEQLEN,
    OUT_EMBEDDINGQUERY,
    OUT_EMBEDDINGKEY,
    OUT_EMBEDDINGVALUE,
    INTERMEDIATE_EMBEDDINGQUERY,
    INTERMEDIATE_EMBEDDINGKEY,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 2;

atb::Status PositionEmbeddingFusionOperation(const PositionEmbedding1dFusionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "positionEmbeddingfusionoperation";

    size_t nodeId = 0;
    auto &splitqkvnode = opGraph.nodes.at(nodeId++);
    auto &ropeNode = opGraph.nodes.at(nodeId++);

    atb::infer::SplitParam splitParam = {1, 3};
    CreateOperation(splitParam, &splitqkvnode.operation);
    splitqkvnode.inTensorIds = {MIXEDQKV};
    splitqkvnode.outTensorIds = {INTERMEDIATE_EMBEDDINGQUERY, INTERMEDIATE_EMBEDDINGKEY, OUT_EMBEDDINGVALUE};
    splitqkvnode.inTensorReshapeFuncs.resize(splitqkvnode.inTensorIds.size());
    splitqkvnode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2维
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维
        newShape.dims[1] = oldShape.dims[2]; // 第二维
    };

    atb::infer::RopeParam ropeparam;
    ropeparam.rotaryCoeff = param.rotaryCoeff;
    CreateOperation(ropeparam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMEDIATE_EMBEDDINGQUERY, INTERMEDIATE_EMBEDDINGKEY,
                            IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {OUT_EMBEDDINGQUERY, OUT_EMBEDDINGKEY};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2维
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维
        newShape.dims[1] = oldShape.dims[2]; // 第二维
    };
    ropeNode.inTensorReshapeFuncs.at(3) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2维
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 第一维
        newShape.dims[1] = oldShape.dims[2]; // 第二维
    };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0); // 0
        outTensorDescs.at(0).shape.dimNum = 4;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = param.headNum;
        outTensorDescs.at(0).shape.dims[3] = inTensorDescs.at(0).shape.dims[2] / param.headNum / 3;
        outTensorDescs.at(1) = inTensorDescs.at(0); // 0
        outTensorDescs.at(1).shape.dimNum = 2;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] * inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[2] / 3;
        outTensorDescs.at(2) = outTensorDescs.at(1); // 1
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed