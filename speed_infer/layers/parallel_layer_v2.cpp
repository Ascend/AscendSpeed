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
#include "nlohmann/json.hpp"

#include "parallel_layer_v2.h"

namespace atb_speed {
namespace common {

enum ParallelType : int {
    ROW_PARALLEL = 0,
    COLUMN_PARALLEL,
};

enum InTensorId : int {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_BIAS,
    IN_DEQSCALE,
    IN_INDEX_IDS,
    OUT_LINEAR,
    INTER_ID,
};

atb::Status ParallelLinearBaseV2(const ParallelParamV2 &param_, atb::Operation **operation,
                                 const ParallelType parallelType)
{
    atb::GraphParam opGraph;

    opGraph.inTensorNum = 5;
    opGraph.outTensorNum = 1;

    // 判断node个数
    size_t nodeCount = 1;
    size_t internalTensorNum = 0;
    if (param_.isQuant) {
        if (param_.quantParam.isQuantOp) {
            nodeCount += 1;
            internalTensorNum += 1;
        }
    } else {
        if (param_.isBias) {
            nodeCount += 1;
        }
    }

    if (param_.commParam.rankSize > 1) {
        nodeCount += 1;
        if (param_.isBias && !param_.isQuant) {
            internalTensorNum += 2;
        } else {
            internalTensorNum += 1;
        }
    }

    opGraph.internalTensorNum = internalTensorNum;
    opGraph.nodes.resize(nodeCount);

    size_t nodeId = 0;
    size_t inter_id = INTER_ID;

    if (!param_.isQuant) {
        atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam matmulParam = {param_.transposeA, param_.transposeB, false};
        atb::CreateOperation(matmulParam, &matmulNode.operation);
        matmulNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        matmulNode.outTensorIds = {(param_.commParam.rankSize > 1 || param_.isBias) ? inter_id : OUT_LINEAR};
    } else {
        if (param_.quantParam.isQuantOp) {
            atb::Node &quantNode = opGraph.nodes.at(nodeId++);
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = param_.quantParam.elewiseType;
            quantParam.quantParam.inputScale = param_.quantParam.inputScale;
            quantParam.quantParam.inputOffset = param_.quantParam.inputOffset;
            atb::CreateOperation(quantParam, &quantNode.operation);
            quantNode.inTensorIds = {IN_INPUT};
            quantNode.outTensorIds = {inter_id};
        }

        atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearQuantParam matmulParam = {param_.transposeA, param_.transposeB, true};
        atb::CreateOperation(matmulParam, &matmulNode.operation);
        matmulNode.inTensorIds = {param_.quantParam.isQuantOp ? inter_id++ : IN_INPUT, IN_WEIGHT, IN_BIAS, IN_DEQSCALE};
        matmulNode.outTensorIds = {param_.commParam.rankSize > 1 ? inter_id : OUT_LINEAR};
    }

    if (param_.commParam.rankSize > 1) {
        atb::Node &parallelNode = opGraph.nodes.at(nodeId++);

        if (parallelType == ROW_PARALLEL) {
            atb::infer::AllReduceParam allReduceParam;
            allReduceParam.rank = param_.commParam.rank;
            allReduceParam.rankSize = param_.commParam.rankSize;
            allReduceParam.backend = param_.commParam.backend;
            atb::CreateOperation(allReduceParam, &parallelNode.operation);
        } else {
            atb::infer::AllGatherParam allGatherParam;
            allGatherParam.rank = param_.commParam.rank;
            allGatherParam.rankSize = param_.commParam.rankSize;
            allGatherParam.backend = param_.commParam.backend;
            atb::CreateOperation(allGatherParam, &parallelNode.operation);
        }

        parallelNode.inTensorIds = {inter_id++};
        parallelNode.outTensorIds = {param_.isBias && !param_.isQuant ? inter_id : OUT_LINEAR};
    }

    if (param_.isBias && !param_.isQuant) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        atb::CreateOperation(addParam, &addNode.operation);
        addNode.inTensorIds = {inter_id, IN_BIAS};
        addNode.outTensorIds = {OUT_LINEAR};
    }

    if (parallelType == ROW_PARALLEL) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            if (param_.isQuant) {
                outTensorDescs.at(0).dtype = ACL_FLOAT16;
            } else {
                outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            }
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            auto w_dim = inTensorDescs.at(1).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            if (param_.isQuant) {
                if (dimNum == 3) {
                    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
                }
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[w_dim-2]; // ND,NZ统一为-2轴
            } else {
                if (dimNum == 3) {
                    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
                }
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[0];
            }
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            if (param_.isQuant) {
                outTensorDescs.at(0).dtype = ACL_FLOAT16;
            } else {
                outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            }
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum + 1; // add rank dim
            outTensorDescs.at(0).shape.dims[0] = param_.commParam.rankSize;
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[0]; // last dim

            return atb::NO_ERROR;
        };
    }

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status ParallelLinearV2(const ParallelParamV2 &param_, atb::Operation **operation, const ParallelType parallelType)
{
    return ParallelLinearBaseV2(param_, operation, parallelType); // 5:in 1:out 3:inter
}


atb::Status RowParallelLinearV2(const ParallelParamV2 &param_, atb::Operation **operation)
{
    return ParallelLinearV2(param_, operation, ROW_PARALLEL);
}

atb::Status ColumnParallelLinearV2(const ParallelParamV2 &param_, atb::Operation **operation)
{
    return ParallelLinearV2(param_, operation, COLUMN_PARALLEL);
}

atb::Status VocabParallelEmbeddingV2(const ParallelParamV2 &param_, atb::Operation **operation) { return 0; }
} // namespace common
} // namespace atb_speed

