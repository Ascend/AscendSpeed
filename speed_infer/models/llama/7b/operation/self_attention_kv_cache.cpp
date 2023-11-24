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
#include "self_attention.h"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include <cmath>

namespace atb_speed {
namespace llama_7b {
static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 13;
enum SelfAttentionKvCacheTensorId {
    IN_MIXED_QUERY = 0,
    IN_MIXED_KEY,
    IN_MIXED_VALUE,
    IN_ATTENTION_MASK,
    IN_PAST_KEY,
    IN_PAST_VALUE,
    OUT_CONTEXT_OUT,
    OUT_PRESENT_KEY,
    OUT_PRESENT_VALUE,
    INTERNAL_Q_SCALED_OUT,
    INTERNAL_TRANSPOSED_Q,
    INTERNAL_TRANSPOSED_K,
    INTERNAL_BMM_Q_K_OUT,
    INTERNAL_ATTENTION_SCORES,
    INTERNAL_ATTENTION_SCORES_F32,
    INTERNAL_ATTENTION_PROBS_F32,
    INTERNAL_ATTENTION_PROBS,
    INTERNAL_TRANSPOSED_V,
    INTERNAL_BMM_V_OUT,
};

atb::Status SelfAttentionKvCache(const SelfAttentionKvCacheParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, transKey:" << param.transKey << ", dk: " << param.dk
                  << ", headNum: " << param.headNum << ", layerId: " << param.layerId;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    auto &catKeyNode = opGraph.nodes.at(nodeId++);
    auto &catValueNode = opGraph.nodes.at(nodeId++);
    auto &mulsQNode = opGraph.nodes.at(nodeId++);
    auto &permuteQNode = opGraph.nodes.at(nodeId++);
    auto &permuteKNode = opGraph.nodes.at(nodeId++);
    auto &bmmQKNode = opGraph.nodes.at(nodeId++);
    auto &addMaskNode = opGraph.nodes.at(nodeId++);
    auto &castInNode = opGraph.nodes.at(nodeId++);
    auto &softMaxNode = opGraph.nodes.at(nodeId++);
    auto &castOutNode = opGraph.nodes.at(nodeId++);
    auto &permuteVNode = opGraph.nodes.at(nodeId++);
    auto &bmmVNode = opGraph.nodes.at(nodeId++);
    auto &transposeContext1Node = opGraph.nodes.at(nodeId++);

    atb::infer::ConcatParam concatParam = {1};
    CreateOperation(concatParam, &catKeyNode.operation);
    catKeyNode.inTensorIds = {IN_PAST_KEY, IN_MIXED_KEY};
    catKeyNode.outTensorIds = {OUT_PRESENT_KEY};

    CreateOperation(concatParam, &catValueNode.operation);
    catValueNode.inTensorIds = {IN_PAST_VALUE, IN_MIXED_VALUE};
    catValueNode.outTensorIds = {OUT_PRESENT_VALUE};

    // scaling down q
    float scalingAttr = 1.0 / sqrt(param.dk);
    ATB_LOG(INFO) << "Scaling down for query with scaling factor " << scalingAttr;
    atb::infer::ElewiseParam scalingElewiseMulsParam;
    scalingElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    scalingElewiseMulsParam.mulsParam.varAttr = scalingAttr;
    CreateOperation(scalingElewiseMulsParam, &mulsQNode.operation);
    mulsQNode.inTensorIds = {IN_MIXED_QUERY};
    mulsQNode.outTensorIds = {INTERNAL_Q_SCALED_OUT};

    atb::infer::TransposeParam permuteSeqHnParam = {{0, 2, 1, 3}};
    CreateOperation(permuteSeqHnParam, &permuteQNode.operation);
    permuteQNode.inTensorIds = {INTERNAL_Q_SCALED_OUT};
    permuteQNode.outTensorIds = {INTERNAL_TRANSPOSED_Q};

    // trans [bs, sq, hn, hs] to [bs, hn, hs, sq]
    atb::infer::TransposeParam permuteSeqHnHsParam = {{0, 2, 3, 1}};
    CreateOperation(permuteSeqHnHsParam, &permuteKNode.operation);
    permuteKNode.inTensorIds = {OUT_PRESENT_KEY};
    permuteKNode.outTensorIds = {INTERNAL_TRANSPOSED_K};

    atb::infer::MatmulParam matmulParam = {false, false};
    CreateOperation(matmulParam, &bmmQKNode.operation);
    bmmQKNode.inTensorIds = {INTERNAL_TRANSPOSED_Q, INTERNAL_TRANSPOSED_K};
    bmmQKNode.outTensorIds = {INTERNAL_BMM_Q_K_OUT};
    bmmQKNode.inTensorReshapeFuncs.resize(bmmQKNode.inTensorIds.size());
    bmmQKNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };
    bmmQKNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &addMaskNode.operation);
    addMaskNode.inTensorIds = {IN_ATTENTION_MASK, INTERNAL_BMM_Q_K_OUT};
    addMaskNode.outTensorIds = {INTERNAL_ATTENTION_SCORES};
    addMaskNode.inTensorReshapeFuncs.resize(addMaskNode.inTensorIds.size());
    addMaskNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0] / param.headNum;
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CreateOperation(castParam, &castInNode.operation);
    castInNode.inTensorIds = {INTERNAL_ATTENTION_SCORES};
    castInNode.outTensorIds = {INTERNAL_ATTENTION_SCORES_F32};

    atb::infer::SoftmaxParam softmaxParam = {{-1}};
    CreateOperation(softmaxParam, &softMaxNode.operation);
    softMaxNode.inTensorIds = {INTERNAL_ATTENTION_SCORES_F32};
    softMaxNode.outTensorIds = {INTERNAL_ATTENTION_PROBS_F32};

    CreateOperation(castParam, &castOutNode.operation);
    castOutNode.inTensorIds = {INTERNAL_ATTENTION_PROBS_F32};
    castOutNode.outTensorIds = {INTERNAL_ATTENTION_PROBS};

    CreateOperation(permuteSeqHnParam, &permuteVNode.operation);
    permuteVNode.inTensorIds = {OUT_PRESENT_VALUE};
    permuteVNode.outTensorIds = {INTERNAL_TRANSPOSED_V};

    CreateOperation(matmulParam, &bmmVNode.operation);
    bmmVNode.inTensorIds = {INTERNAL_ATTENTION_PROBS, INTERNAL_TRANSPOSED_V};
    bmmVNode.outTensorIds = {INTERNAL_BMM_V_OUT};
    bmmVNode.inTensorReshapeFuncs.resize(bmmVNode.inTensorIds.size());
    bmmVNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };
    bmmVNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
    };

    CreateOperation(permuteSeqHnParam, &transposeContext1Node.operation);
    transposeContext1Node.inTensorIds = {INTERNAL_BMM_V_OUT};
    transposeContext1Node.outTensorIds = {OUT_CONTEXT_OUT};
    transposeContext1Node.inTensorReshapeFuncs.resize(transposeContext1Node.inTensorIds.size());
    transposeContext1Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4;
        newShape.dims[0] = oldShape.dims[0] / param.headNum;
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1];
        newShape.dims[3] = oldShape.dims[2];
    };

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[2] * inTensorDescs[0].shape.dims[3];

        outTensorDescs.at(1) = inTensorDescs.at(4);
        outTensorDescs.at(1).shape.dims[1] = outTensorDescs.at(1).shape.dims[1] + 1;
        outTensorDescs.at(2) = inTensorDescs.at(5);
        outTensorDescs.at(2).shape.dims[1] = outTensorDescs.at(2).shape.dims[1] + 1;

        ATB_LOG(INFO) << __func__ << " infer shape success";
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed