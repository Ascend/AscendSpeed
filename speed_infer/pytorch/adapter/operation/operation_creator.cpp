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
#include <functional>
#include <nlohmann/json.hpp>
#include "atb_speed/log.h"
#include "operation_creator.h"

using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &paramJson)>;

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    } 
    ATB_LOG(INFO) << "AllReduceParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllReduceParam rankSize:" << param.rankSize;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    } 
    ATB_LOG(INFO) << "AllGatherParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllGatherParam rankSize:" << param.rankSize;
    ATB_LOG(INFO) << "AllGatherParam backend:" << param.backend;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *BroadcastOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::BroadcastParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    ATB_LOG(INFO) << "BroadcastParam rank:" << param.rank << "rankSize:" << param.rankSize
                  << "rankRoot:" << param.rankRoot;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *PositionEmbedding1dSplitFusionOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *AddNormOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *EmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *NormOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ATB_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TransdataOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransdataParam param;
    param.transdataType = atb::infer::TransdataParam::ND_TO_FRACTAL_NZ;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}


static atb::Operation *FfnOldOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearActivationParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType = atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "FfnParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}


static atb::Operation *MlpOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *MlpQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *PositionEmbedding1dSplitOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *PositionEmbeddingOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *SelfAttentionKvCacheOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "transpose(" << param.perm << ")";
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}


static atb::Operation *LinearActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearActivationParam param;
    param.transposeA = paramJson["transposeA"].get<bool>();
    param.transposeB = paramJson["transposeB"].get<bool>();
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType = atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "LinearActivationParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *AddNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *NormQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *QuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *SelfAttentionKvCacheFusionOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *WordEmbeddingParallelOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *LmHeadParallelOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *RmsPreNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *RmsNormQuantOperationCreate(const nlohmann::json &paramJson)
{
    return nullptr;
}

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType =  atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    ATB_LOG(INFO) << "ActivationParam activationType:" << param.activationType << ", scale:" << param.scale;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"RmsPreNormQuantOperation", &RmsPreNormQuantOperationCreate},
    {"RmsNormQuantOperation", &RmsNormQuantOperationCreate},
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"NormOperation", &NormOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"PositionEmbedding1dSplitFusionOperation", &PositionEmbedding1dSplitFusionOperationCreate},
    {"AddNormOperation", &AddNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"TransdataOperation", &TransdataOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"FfnOldOperation", &FfnOldOperationCreate},
    {"MlpOperation", &MlpOperationCreate},
    {"MlpQuantOperation", &MlpQuantOperationCreate},
    {"EmbeddingOperation", &EmbeddingOperationCreate},
    {"PositionEmbedding1dSplitOperation", &PositionEmbedding1dSplitOperationCreate},
    {"PositionEmbedding", &PositionEmbeddingOperationCreate},
    {"SelfAttentionKvCache", &SelfAttentionKvCacheOperationCreate},
    {"SelfAttentionKvCacheFusionOperation", &SelfAttentionKvCacheFusionOperationCreate},
    {"SelfAttention", &SelfAttentionOperationCreate},
    {"QuantOperation", &QuantOperationCreate},
    {"AddNormQuantOperation", &AddNormQuantOperationCreate},
    {"NormQuantOperation", &NormQuantOperationCreate},
    {"LinearQuantOperation", &LinearQuantOperationCreate},
    {"LinearActivationOperation", &LinearActivationOperationCreate},
    {"LmHeadParallelOperation", &LmHeadParallelOperationCreate},
    {"WordEmbeddingParallelOperation", &WordEmbeddingParallelOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
};

atb::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}