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
#ifndef ATB_SPEED_MODELS_LLAMA_7B_SELF_ATTETNTION_H
#define ATB_SPEED_MODELSLLAMA_7B_SELF_ATTETNTION_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama_7b {

struct SelfAttentionKvCacheParam {
    bool transKey = false;
    int64_t dk = 0;
    int64_t headNum = 0;
    int64_t layerId = 0;
    float preScale = 0;
    float postScale = 0;
    int64_t numHeadsPerPartition = 0;
    int64_t hiddenSizePerHead = 0;
    int64_t numGroupsPerPartition = 0;
    std::string model = "llama_7b";
    float invNormFactorVarAttr = 0;
};
struct SelfAttentionParam {
    bool transKey = false;
    int64_t dk = 0;
    int64_t headNum = 0;
    int64_t layerId = 0;
    float preScale = 0;
    float postScale = 0;
    int64_t numHeadsPerPartition = 0;
    int64_t hiddenSizePerHead = 0;
    int64_t numGroupsPerPartition = 0;
    std::string model = "llama_7b";
};

atb::Status SelfAttentionKvCache(const SelfAttentionKvCacheParam &param, atb::Operation **operation);

atb::Status SelfAttention(const SelfAttentionParam &param, atb::Operation **operation);

static atb::Operation *CreateSelfAttention(const nlohmann::json &paramJson)
{
    SelfAttentionParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    ATB_LOG(INFO) << "LLaMA_7B_SelfAttentionParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead;
    atb::Operation *op;
    SelfAttention(param, &op);
    return op;
}

static atb::Operation *CreateSelfAttentionKVCache(const nlohmann::json &paramJson)
{
    SelfAttentionKvCacheParam param;
    if (paramJson.contains("transKey")) {
        param.transKey = paramJson["transKey"].get<bool>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("layerId")) {
        param.layerId = paramJson["layerId"].get<int>();
    }
    if (paramJson.contains("dk")) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.contains("model")) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("preScale")) {
        param.preScale = paramJson["preScale"].get<float>();
    }
    if (paramJson.contains("postScale")) {
        param.postScale = paramJson["postScale"].get<float>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int64_t>();
    }
    if (paramJson.contains("hiddenSizePerHead")) {
        param.hiddenSizePerHead = paramJson["hiddenSizePerHead"].get<int64_t>();
    }
    if (paramJson.contains("numGroupsPerPartition")) {
        param.numGroupsPerPartition = paramJson["numGroupsPerPartition"].get<int64_t>();
    }
    ATB_LOG(INFO) << "LLaMA_7B_SelfAttentionKvCacheParam transKey:" << param.transKey << ", headNum:" << param.headNum
                  << ", layerId:" << param.layerId << ", dk:" << param.dk << ", preScale" << param.preScale
                  << ", postScale" << param.postScale << ", model" << param.model << ", hiddenSizePerHead"
                  << param.hiddenSizePerHead;
    atb::Operation *op;
    SelfAttentionKvCache(param, &op);
    return op;
}
} // namespace llama_7b

} // namespace atb_speed
#endif