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
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

#include "models/llama/13b/layer/parallel_layer.h"

#include "rope_model.h"

namespace atb_speed {
namespace llama_7b {
const int WEIGHT_COUNT_PER_LAYER = 9;

enum InTensorId {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_END
};

void EncoderRopeModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    ATB_LOG(INFO) << "Llama7BEncoderRopeModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize" << rankSize;
}

EncoderRopeModel::EncoderRopeModel(const std::string &param) : Model("EncoderRopeModel", param)
{
    param_.FromString(param);
}

EncoderRopeModel::~EncoderRopeModel() = default;

uint32_t EncoderRopeModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t EncoderRopeModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status EncoderRopeModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                         std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter EncoderRopeModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const atb::TensorDesc &keyTensorDesc = inTensorDescs.at(0);

    outTensorDescs.at(0) = inTensorDescs.at(0);
    ATB_LOG(INFO) << "EncoderRopeModel InferShape Looping";
    for (size_t keyId = 0; keyId < param_.layerNum; ++keyId) {
        outTensorDescs.at(1 + keyId) = keyTensorDesc;
        outTensorDescs.at(1 + keyId).shape.dimNum += 1;
        outTensorDescs.at(1 + keyId).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1 + keyId).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1 + keyId).shape.dims[2] = param_.headNum;
        outTensorDescs.at(1 + keyId).shape.dims[3] = param_.dk;
    }
    for (size_t valueId = 0; valueId < param_.layerNum; ++valueId) {
        outTensorDescs.at(1 + param_.layerNum + valueId) = outTensorDescs.at(1 + valueId);
    }

    return atb::NO_ERROR;
}

void EncoderRopeModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter EncoderRopeModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_PER_LAYER * param_.layerNum;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_END);
    graph_.outTensors.resize(1 + param_.layerNum * 2);

    const int nodeSize = param_.layerNum;
    graph_.nodes.resize(nodeSize);

    graph_.internalTensors.resize(graph_.nodes.size() - 1);

    int nodeId = 0;
    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);
    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::llama_13b::ParallelLayerParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.model = "llama13b";
        modelParam.rank = param_.rank;
        modelParam.rankSize = param_.rankSize;
        atb_speed::llama_13b::EncoderParallelRopeLayer(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor; // hidden states
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);    // positionIdTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);      // cosTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);      // sinTable
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK); // attentionMaskTensor
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN);

        if (layerId != param_.layerNum - 1) {
            layerNode.outTensors = {&graph_.internalTensors.at(layerId), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        } else {
            layerNode.outTensors = {&graph_.outTensors.at(0), &graph_.outTensors.at(layerId + 1),
                                    &graph_.outTensors.at(layerId + 1 + param_.layerNum)};
        }

        firstInTensor = layerNode.outTensors.at(0);
    }
}
} // namespace llama_7b
} // namespace atb_speed