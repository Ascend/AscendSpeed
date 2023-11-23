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
#ifndef ATB_SPEED_MODELS_LLAMA_13B_LAYER_PARALLEL_H
#define ATB_SPEED_MODELS_LLAMA_13B_LAYER_PARALLEL_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace llama_13b {
struct ParallelLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama_13b";
};

atb::Status EncoderParallelLayer(const ParallelLayerParam &param, atb::Operation **operation);

atb::Status DecoderParallelLayer(const ParallelLayerParam &param, atb::Operation **operation);

atb::Status EncoderParallelRopeLayer(const ParallelLayerParam &param, atb::Operation **operation);

atb::Status DecoderParallelRopeLayer(const ParallelLayerParam &param, atb::Operation **operation);

static atb::Operation *CreateEncoderParallelLayer(const nlohmann::json &paramJson)
{
    ParallelLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA13BLayerEncoder headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize"
                  << param.rankSize;
    atb::Operation *op;
    EncoderParallelLayer(param, &op);
    return op;
}

static atb::Operation *CreateDecoderParallelLayer(const nlohmann::json &paramJson)
{
    ParallelLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA13BLayerEncoder headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize"
                  << param.rankSize;
    atb::Operation *op;
    DecoderParallelLayer(param, &op);
    return op;
}

static atb::Operation *CreateEncoderParallelRopeLayer(const nlohmann::json &paramJson)
{
    ParallelLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA13BLayerEncoderRope headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize"
                  << param.rankSize;
    atb::Operation *op;
    EncoderParallelRopeLayer(param, &op);
    return op;
}

static atb::Operation *CreateDecoderParallelRopeLayer(const nlohmann::json &paramJson)
{
    ParallelLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    ATB_LOG(INFO) << "LLaMA13BLayerEncoderRope headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize"
                  << param.rankSize;
    atb::Operation *op;
    DecoderParallelRopeLayer(param, &op);
    return op;
}
} // namespace llama_13b
} // namespace atb_speed
#endif