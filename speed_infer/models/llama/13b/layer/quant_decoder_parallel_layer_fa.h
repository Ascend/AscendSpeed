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
#ifndef LLAMA_13B_QUANT_LAYER_FLASHATTENTION_OPERATION_H
#define LLAMA_13B_QUANT_LAYER_FLASHATTENTION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace llama_13b {
struct QuantLayerParallelFlashAttentionParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0; // headDim
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama_13b";
    float qScale = 1.0;
    // 量化参数
    float qkvInputScale = 1;
    int qkvInputOffset = 0;
    float denseInputScale = 1;
    int denseInputOffset = 0;
    float selfLnInputScale = 1;
    int selfLnInputOffset = 0;
    float ffnOutInputScale = 1;
    int ffnOutInputOffset = 0;
};

atb::Status QuantLayerParallelFlashAttentionOperation(const QuantLayerParallelFlashAttentionParam &param,
    atb::Operation **operation);

static atb::Operation *CreateQuantLayerParallelFlashAttentionOperation(const nlohmann::json &paramJson)
{
    QuantLayerParallelFlashAttentionParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    // 量化参数
    param.qkvInputScale = paramJson["qkvInputScale"].get<float>();
    param.qkvInputOffset = paramJson["qkvInputOffset"].get<int>();
    param.denseInputScale = paramJson["denseInputScale"].get<float>();
    param.denseInputOffset = paramJson["denseInputOffset"].get<int>();
    param.selfLnInputScale = paramJson["selfLnInputScale"].get<float>();
    param.selfLnInputOffset = paramJson["selfLnInputOffset"].get<int>();
    param.ffnOutInputScale = paramJson["ffnOutInputScale"].get<float>();
    param.ffnOutInputOffset = paramJson["ffnOutInputOffset"].get<int>();

    ATB_LOG(INFO) << "LLaMA13BLayerEncoder headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize:"
                  << param.rankSize;
    atb::Operation *op;
    QuantLayerParallelFlashAttentionOperation(param, &op);
    return op;
}

class QuantLayerPrallelFlashAttentionBinder : public HostTensorBinder {
public:
    QuantLayerPrallelFlashAttentionBinder();
    virtual ~QuantLayerPrallelFlashAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
};

} // namespace llama_13b
} // namespace atb_speed
#endif