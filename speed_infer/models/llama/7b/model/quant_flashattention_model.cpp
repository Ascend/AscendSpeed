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
#include "flashattention_model.h"
#include "models/llama/13b/layer/quant_decoder_parallel_layer_fa.h"
#include "models/llama/13b/layer/decoder_parallel_layer_fa.h"
#include "quant_flashattention_model.h"

namespace atb_speed {
namespace llama_7b {
const int WEIGHT_COUNT_PER_LAYER = 23;
const int ROLLBACK_WEIGHT_COUNT_PER_LAYER = 9;
const int INPUT_TENSOR_COUNT_BEFORE_KEY = 11;
const int OUTPUT_TENSOR_COUNT_BEFORE_KEY = 1;
const int WORDEMBEDDINGNODE_WEIGHT_COUNT = 1;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int OPERATION_COUNT_BEFORE_LAYER = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 1;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int MODEL_OUT_DIM_NUM = 3;
const int MODEL_OUT_DIM0 = 0;
const int MODEL_OUT_DIM1 = 1;
const int MODEL_OUT_DIM2 = 2;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_POSITIONID,
    IN_TENSOR_COSTABLE,
    IN_TENSOR_SINTABLE,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_PAST_KEY,
    IN_TENSOR_PAST_VALUE,
    IN_TENSOR_TOKENOFFSET,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_BETA,
    IN_HOLDER,
    IN_TENSOR_MAX, // 9
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void QuantFlashAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();

    for (auto item : paramJson["qkvInputScale"]) {
        qkvInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["qkvInputOffset"]) {
        qkvInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["denseInputScale"]) {
        denseInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["denseInputOffset"]) {
        denseInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["selfLnInputScale"]) {
        selfLnInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["selfLnInputOffset"]) {
        selfLnInputOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["ffnOutInputScale"]) {
        ffnOutInputScale.push_back(item.get<float>());
    }
    for (auto item : paramJson["ffnOutInputOffset"]) {
        ffnOutInputOffset.push_back(item.get<int>());
    }
    floatLayer = paramJson["floatLayer"].get<int>();

    ATB_LOG(INFO) << "Llama QuantFlashAttentionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", rank:" << rank << ", rankSize:" << rankSize;
}

QuantFlashAttentionModel::QuantFlashAttentionModel(const std::string &param) : Model("QuantFlashAttentionModel", param)
{
    param_.FromString(param);
}

QuantFlashAttentionModel::~QuantFlashAttentionModel() {}

uint32_t QuantFlashAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t QuantFlashAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status QuantFlashAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                                 std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter LLaMA QuantFlashAttentionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(MODEL_OUT_DIM0) = graph_.weightTensors.at(MODEL_OUT_DIM0).desc;
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dimNum = MODEL_OUT_DIM_NUM;
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM0] =
        inTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM0];
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM1] =
        inTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM1];
    outTensorDescs.at(MODEL_OUT_DIM0).shape.dims[MODEL_OUT_DIM2] = outDim;

    ATB_LOG(INFO) << "LLaMA QuantFlashAttentionModel InferShape Success";
    return atb::NO_ERROR;
}

void QuantFlashAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter LLaMA QuantFlashAttentionModel BuildGraph";

    const int weightTensorSize = WORDEMBEDDINGNODE_WEIGHT_COUNT +
                                 ROLLBACK_WEIGHT_COUNT_PER_LAYER + WEIGHT_COUNT_PER_LAYER * (param_.layerNum - 1) +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.inTensors.resize(IN_TENSOR_MAX + param_.layerNum);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = param_.layerNum + OPERATION_COUNT_BEFORE_LAYER + OPERATION_COUNT_AFTER_LAYER;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    auto &wordEmbeddingNode = graph_.nodes.at(nodeId++);
    atb::infer::GatherParam wordEmbeddingParam;
    atb::Operation *op = nullptr;
    atb::CreateOperation(wordEmbeddingParam, &op);
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {&graph_.weightTensors.at(0), &graph_.inTensors.at(0)};
    wordEmbeddingNode.outTensors = {&graph_.internalTensors.at(0)};

    atb::Tensor *firstInTensor = &graph_.internalTensors.at(0);

    for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        if (layerId == param_.floatLayer) {
            atb_speed::llama_13b::LayerParallelFlashAttentionParam modelParamRollback;
            modelParamRollback.rmsNormEps = param_.rmsNormEps;
            modelParamRollback.headNum = param_.headNum;
            modelParamRollback.dk = param_.dk;
            modelParamRollback.model = "llama13b";
            modelParamRollback.rank = param_.rank;
            modelParamRollback.rankSize = param_.rankSize;

            atb_speed::llama_13b::LayerParallelFlashAttentionOperation(modelParamRollback, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            for (size_t weightTensorId = 0; weightTensorId < ROLLBACK_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                    layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        } else {
            atb_speed::llama_13b::QuantLayerParallelFlashAttentionParam modelParam;
            modelParam.rmsNormEps = param_.rmsNormEps;
            modelParam.headNum = param_.headNum;
            modelParam.dk = param_.dk;
            modelParam.model = "llama13b";
            modelParam.rank = param_.rank;
            modelParam.rankSize = param_.rankSize;
            // 量化适配
            modelParam.qkvInputScale = param_.qkvInputScale[layerId];
            modelParam.qkvInputOffset = param_.qkvInputOffset[layerId];
            modelParam.denseInputScale = param_.denseInputScale[layerId];
            modelParam.denseInputOffset = param_.denseInputOffset[layerId];
            modelParam.selfLnInputScale = param_.selfLnInputScale[layerId];
            modelParam.selfLnInputOffset = param_.selfLnInputOffset[layerId];
            modelParam.ffnOutInputScale = param_.ffnOutInputScale[layerId];
            modelParam.ffnOutInputOffset = param_.ffnOutInputOffset[layerId];

            atb_speed::llama_13b::QuantLayerParallelFlashAttentionOperation(modelParam, &op);
            layerNode.operation.reset(op);
            layerNode.inTensors.resize(layerNode.operation->GetInputNum());
            layerNode.outTensors.resize(layerNode.operation->GetOutputNum());

            size_t inTensorId = 0;
            layerNode.inTensors.at(inTensorId++) = firstInTensor;
            int weightOffset = 0;
            if (layerId < param_.floatLayer) {
                for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                        layerId * WEIGHT_COUNT_PER_LAYER + weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
                }
            } else {
                for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
                    layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                        ROLLBACK_WEIGHT_COUNT_PER_LAYER + (layerId - 1) * WEIGHT_COUNT_PER_LAYER +
                        weightTensorId + WORDEMBEDDINGNODE_WEIGHT_COUNT);
                }
            }
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_POSITIONID);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_COSTABLE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SINTABLE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_KEY);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_PAST_VALUE);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_TOKENOFFSET);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BETA);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_HOLDER);
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_MAX + layerId);

            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};

            firstInTensor = layerNode.outTensors.at(0);
        }
    }

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    atb::CreateOperation(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};

    auto &outLinearNode = graph_.nodes.at(nodeId++);
    atb::infer::LinearParam outLinearParm = {false, false, false};
    atb::CreateOperation(outLinearParm, &op);
    outLinearNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    outLinearNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                               &graph_.weightTensors.at(finalLinearWeightTensorId)};
    outLinearNode.outTensors = {&graph_.outTensors.at(0)};

    ATB_LOG(INFO) << "LLaMA QuantFlashAttentionModel BuildGraph success";
}

atb::Status QuantFlashAttentionModel::ParseParam(const std::string &param)
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

    return atb::NO_ERROR;
}

atb::Status QuantFlashAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);

    const uint32_t floatTokenOffsetTensorId = 16;
    const uint32_t floatSeqLenTensorId = 17;
    const uint32_t quantTokenOffsetTensorId = 30;
    const uint32_t quantSeqLenTensorId = 31;

    int rollBackNode = param_.floatLayer + 1;
    if (nodeId == rollBackNode) {
        node.variantPack.inTensors.at(floatTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(floatSeqLenTensorId).hostData = seqLen_.data();
    } else {
        node.variantPack.inTensors.at(quantTokenOffsetTensorId).hostData = tokenOffset_.data();
        node.variantPack.inTensors.at(quantSeqLenTensorId).hostData = seqLen_.data();
    }

    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}

} // namespace llama_7b
} // namespace atb_speed