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
#include "llama_position_embedding_1d_split_operation.h"
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {

enum LlamaPositionEmbedding1DSplitTensorId {
    IN_INPUTTENSOR = 0,
    IN_POSITIONIDSTENSOR,
    IN_COSTABLETENSOR,
    IN_SINTABLETENSOR,
    OUT_EMBEDDEDPERMUTEDTENSOR,
    INTERMIDATE_INPUT_TRANSPOSED,
    INTERMIDATE_COS,
    INTERMIDATE_SIN,
    INTERMIDATE_INPUT_TRANSPOSED0,
    INTERMIDATE_INPUT_TRANSPOSED1,
    INTERMIDATE_INPUT_TRANSPOSED1NEG,
    INTERMIDATE_INPUT_ROTATE,
    INTERMIDATE_INPUT_MUL0,
    INTERMIDATE_INPUT_MUL1,
    INTERMIDATE_INPUT_EMBEDDED,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 10;
static uint64_t DIM3 = 3;

atb::Status CreateLlamaPositionEmbedding1DSplitOperation(const llamaPositionEmbedding1DSplitParam &param,
                                                         atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputTransposeNode = opGraph.nodes.at(nodeId++);
    atb::Node &embedding0Node = opGraph.nodes.at(nodeId++);
    atb::Node &embedding1Node = opGraph.nodes.at(nodeId++);
    atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
    atb::Node &negNode = opGraph.nodes.at(nodeId++);
    atb::Node &cat0Node = opGraph.nodes.at(nodeId++);
    atb::Node &mul0Node = opGraph.nodes.at(nodeId++);
    atb::Node &mul1Node = opGraph.nodes.at(nodeId++);
    atb::Node &addNode = opGraph.nodes.at(nodeId++);
    atb::Node &permuteNode = opGraph.nodes.at(nodeId++);

    atb::infer::TransposeParam inputTransposeNodeParam = {{0, 2, 1, 3}};
    CreateOperation(inputTransposeNodeParam, &inputTransposeNode.operation);
    inputTransposeNode.inTensorIds = {IN_INPUTTENSOR};
    inputTransposeNode.outTensorIds = {INTERMIDATE_INPUT_TRANSPOSED};
    inputTransposeNode.inTensorReshapeFuncs.resize(inputTransposeNode.inTensorIds.size());
    inputTransposeNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb::infer::GatherParam embedding0NodeParam;
    embedding0NodeParam.axis = 0;
    CreateOperation(embedding0NodeParam, &embedding0Node.operation);
    embedding0Node.inTensorIds = {IN_COSTABLETENSOR, IN_POSITIONIDSTENSOR};
    embedding0Node.outTensorIds = {INTERMIDATE_COS};
    embedding0Node.inTensorReshapeFuncs.resize(embedding0Node.inTensorIds.size());
    embedding0Node.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dims[i + 2];
            }
        } else {
            newShape = oldShape;
        }
    };

    atb::infer::GatherParam embedding1NodeParam;
    embedding1NodeParam.axis = 0;
    CreateOperation(embedding1NodeParam, &embedding1Node.operation);
    embedding1Node.inTensorIds = {IN_SINTABLETENSOR, IN_POSITIONIDSTENSOR};
    embedding1Node.outTensorIds = {INTERMIDATE_SIN};
    embedding1Node.inTensorReshapeFuncs.resize(embedding1Node.inTensorIds.size());
    embedding1Node.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dims[i + 2];
            }
        } else if (oldShape.dims[0] == 1 && oldShape.dims[2] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            newShape.dims[0] = oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[3];
        } else {
            newShape = oldShape;
        }
    };

    atb::infer::SplitParam splitParam = {3, 2};
    CreateOperation(splitParam, &sliceNode.operation);
    sliceNode.inTensorIds = {INTERMIDATE_INPUT_TRANSPOSED};
    sliceNode.outTensorIds = {INTERMIDATE_INPUT_TRANSPOSED0, INTERMIDATE_INPUT_TRANSPOSED1};

    atb::infer::ElewiseParam negNodeParam;
    negNodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_NEG;
    CreateOperation(negNodeParam, &negNode.operation);
    negNode.inTensorIds = {INTERMIDATE_INPUT_TRANSPOSED1};
    negNode.outTensorIds = {INTERMIDATE_INPUT_TRANSPOSED1NEG};

    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = 3;
    CreateOperation(concatParam, &cat0Node.operation);
    cat0Node.inTensorIds = {INTERMIDATE_INPUT_TRANSPOSED1NEG, INTERMIDATE_INPUT_TRANSPOSED0};
    cat0Node.outTensorIds = {INTERMIDATE_INPUT_ROTATE};

    atb::infer::ElewiseParam mul0NodeParam;
    mul0NodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mul0NodeParam, &mul0Node.operation);
    mul0Node.inTensorIds = {INTERMIDATE_INPUT_TRANSPOSED, INTERMIDATE_COS};
    mul0Node.outTensorIds = {INTERMIDATE_INPUT_MUL0};
    mul0Node.inTensorReshapeFuncs.resize(mul0Node.inTensorIds.size());
    // For cos and sin: [batch_size, 1, max_seq_len, head_dims]
    mul0Node.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam mul1NodeParam;
    mul1NodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mul1NodeParam, &mul1Node.operation);
    mul1Node.inTensorIds = {INTERMIDATE_INPUT_ROTATE, INTERMIDATE_SIN};
    mul1Node.outTensorIds = {INTERMIDATE_INPUT_MUL1};
    mul1Node.inTensorReshapeFuncs.resize(mul1Node.inTensorIds.size());
    // For cos and sin: [batch_size, 1, max_seq_len, head_dims]
    mul1Node.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam addNodeParam;
    addNodeParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addNodeParam, &addNode.operation);
    addNode.inTensorIds = {INTERMIDATE_INPUT_MUL0, INTERMIDATE_INPUT_MUL1};
    addNode.outTensorIds = {INTERMIDATE_INPUT_EMBEDDED};

    atb::infer::TransposeParam permuteNodeParam = {{2, 0, 1, 3}};
    CreateOperation(permuteNodeParam, &permuteNode.operation);
    permuteNode.inTensorIds = {INTERMIDATE_INPUT_EMBEDDED};
    permuteNode.outTensorIds = {OUT_EMBEDDEDPERMUTEDTENSOR};
    permuteNode.inTensorReshapeFuncs.resize(permuteNode.inTensorIds.size());

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        // in : Q,[batch, seq_len, all_head_size]   position_ids,[]  cos_table,[]  sin_table[]
        // out : Q ,[seq_len, batch, head_num, head_size]
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 4;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[2] = param.headNum; // 2 ：输出张量第三维的大小，即head_num的值
        outTensorDescs.at(0).shape.dims[3] =
            inTensorDescs.at(0).shape.dims[2] / param.headNum; // 3 ：输出张量第四维的大小，即head_num的值

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed