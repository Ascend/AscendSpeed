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
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/llama/7b/layer/layer_fusion.h"
#include "fusion_model.h"

namespace atb_speed {
namespace llama_7b {
const int WEIGHT_COUNT_PER_LAYER = 6;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_BEFORE_LAYER = 3;
const int OPERATION_COUNT_AFTER_LAYER = 2;  // norm + lm_head

enum InTensorId {
    IN_TENSOR_INPUTIDS = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_MAX,  // 9
};

enum OutTensorId {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void FusionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();

    if (paramJson.contains("rotaryCoeff")) {
        rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    ATB_LOG(INFO) << "Llama FusionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum
                  << ", tokenOffset:" << tokenOffset << ", seqLen:" << seqLen
                  << ", rotaryCoeff:" << rotaryCoeff;
}

FusionModel::FusionModel(const std::string &param) : Model("FusionModel", param)
{
    param_.FromString(param);
}

FusionModel::~FusionModel() = default;

uint32_t FusionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t FusionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status FusionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                    std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter LlamaFusionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    outTensorDescs.at(0) = graph_.weightTensors.at(0).desc;
    outTensorDescs.at(0).shape.dimNum = 3; // 3维
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
    outTensorDescs.at(0).shape.dims[2] = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    return atb::NO_ERROR;
}

void FusionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter LlamaFusionModel BuildGraph";

    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT + WEIGHT_COUNT_PER_LAYER * param_.layerNum +
                                FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;

    graph_.weightTensors.resize(weightTensorSize);
    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);
    
    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() + 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;

    auto &wordembeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordembeddinggatherparam;
    atb::CreateOperation(wordembeddinggatherparam, &op);
    wordembeddingNode.operation.reset(op);
    wordembeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(IN_TENSOR_INPUTIDS)};
    wordembeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    auto &cosNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam cosembeddinggatherparam;
    atb::CreateOperation(cosembeddinggatherparam, &op);
    cosNode.operation.reset(op);
    cosNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_COSTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    cosNode.outTensors = {&graph_.internalTensors.at(1)};

    auto &sinNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam sinembeddinggatherparam;
    atb::CreateOperation(sinembeddinggatherparam, &op);
    sinNode.operation.reset(op);
    sinNode.inTensors = {&graph_.inTensors.at(IN_TENSOR_SINTABLE), &graph_.inTensors.at(IN_TENSOR_POSITIONID)};
    sinNode.outTensors = {&graph_.internalTensors.at(2)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);
    atb::Tensor *cosEmbedTensor = &graph_.internalTensors.at(1);
    atb::Tensor *sinEmbedTensor = &graph_.internalTensors.at(2);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);
        atb_speed::llama_7b::LayerFusionParam modelParam;
        modelParam.rmsNormEps = param_.rmsNormEps;
        modelParam.headNum = param_.headNum;
        modelParam.dk = param_.dk;
        modelParam.layerId = layerId;
        modelParam.tokenOffset = param_.tokenOffset;
        modelParam.rotaryCoeff = param_.rotaryCoeff;
        modelParam.model = "llama7b";

        atb_speed::llama_7b::FusionLayerOperation(modelParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
        }
        layerNode.inTensors.at(inTensorId++) = cosEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = sinEmbedTensor;
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        firstInTensor = layerNode.outTensors.at(0);
    }
    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
    atb::CreateOperation(rmsNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId = graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT
                                            - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam linearParam = {false, false, false};
    atb::CreateOperation(linearParam, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                                &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};
    ATB_LOG(INFO) << "build graph end";
}

atb::Status FusionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }

    ATB_LOG(INFO) << "FusionModel ParseParam tokenOffset:" << tokenOffset_ << ", seqLen:" << seqLen_;
    return atb::NO_ERROR;
}

atb::Status FusionModel::BindParamHostTensor(uint32_t nodeId)
{
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    ATB_LOG(INFO) << "BindParamHostTensor";
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t tokenOffsetTensorId = 12;
    const uint32_t seqLenTensorId = 13;
    node.variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}

} // namespace llama_7b
} // namespace atb_speed