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
#include "parallel_layer.h"

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
enum ParallelType : int {
    ROW_PARALLEL = 0,
    COLUMN_PARALLEL,
};

template <class T>
atb::Status ParallelLinearBase(const ParallelParam &param_, atb::Operation **operation, T config,
                               const ParallelType parallelType)
{
    atb::GraphParam opGraph;

    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);

    size_t nodeId = 0;
    atb::Node &matmulNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulParam = {param_.transposeA, param_.transposeB, false};
    atb::CreateOperation(matmulParam, &matmulNode.operation);
    matmulNode.inTensorIds = {config.IN_INPUT, config.IN_WEIGHT};
    matmulNode.outTensorIds = {config.INTERMIDATE_MATMULOUT};

    if (param_.rankSize > 1) {
        atb::Node &parallelNode = opGraph.nodes.at(nodeId++);

        if (parallelType == ROW_PARALLEL) {
            atb::infer::AllReduceParam allReduceParam;
            allReduceParam.rank = param_.rank;
            allReduceParam.rankSize = param_.rankSize;
            allReduceParam.backend = param_.backend;
            atb::CreateOperation(allReduceParam, &parallelNode.operation);
        } else {
            atb::infer::AllGatherParam allGatherParam;
            allGatherParam.rank = param_.rank;
            allGatherParam.rankSize = param_.rankSize;
            allGatherParam.backend = param_.backend;
            atb::CreateOperation(allGatherParam, &parallelNode.operation);
        }

        parallelNode.inTensorIds = {config.INTERMIDATE_MATMULOUT};
        parallelNode.outTensorIds = {config.INTERMIDATE_ALLREDUCEOUT};
    }

    if (param_.isBias) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        atb::CreateOperation(addParam, &addNode.operation);
        addNode.inTensorIds = {param_.rankSize > 1 ? config.INTERMIDATE_ALLREDUCEOUT : config.INTERMIDATE_MATMULOUT,
                               config.IN_BIAS};
        addNode.outTensorIds = {config.OUT_LINEAROUT};
    }

    if (parallelType == ROW_PARALLEL) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            }
            if (param_.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[1];
            } else {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[0];
            }

            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                     atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum + 1; // add rank dim
            outTensorDescs.at(0).shape.dims[0] = param_.rankSize;
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            if (param_.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[1]; // last dim
            } else {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[0]; // last dim
            }

            return atb::NO_ERROR;
        };
    }

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status ParallelLinear(const ParallelParam &param_, atb::Operation **operation, const ParallelType parallelType)
{
    if (param_.isBias && (param_.rankSize > 1)) {
        return ParallelLinearBase(param_, operation, LinearWithBiasAndParallel(3, 1, 2, 3),
                                  parallelType); // 3:in 1:out 2:inter 3:node
    } else if (param_.isBias) {
        return ParallelLinearBase(param_, operation, LinearWithBias(3, 1, 1, 2),
                                  parallelType); // 3:in 1:out 1:inter 2:node
    } else if (param_.rankSize > 1) {
        return ParallelLinearBase(param_, operation, LinearWithParallel(2, 1, 1, 2),
                                  parallelType); // 2:in 1:out 1:inter 2:node
    } else {
        return ParallelLinearBase(param_, operation, LinearOnly(2, 1, 0, 1), parallelType); // 2:in 1:out 0:inter 1:node
    }
}

atb::Status RowParallelLinear(const ParallelParam &param_, atb::Operation **operation)
{
    return ParallelLinear(param_, operation, ROW_PARALLEL);
}

atb::Status ColumnParallelLinear(const ParallelParam &param_, atb::Operation **operation)
{
    return ParallelLinear(param_, operation, COLUMN_PARALLEL);
}

atb::Status VocabParallelEmbedding(const ParallelParam &param_, atb::Operation **operation) { return 0; }

} // namespace common
} // namespace atb_speed
