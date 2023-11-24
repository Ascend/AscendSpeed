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
#ifndef LLAMA_13B_LAYER_FLASHATTENTION_OPERATION_H
#define LLAMA_13B_LAYER_FLASHATTENTION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "nlohmann/json.hpp"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace llama_13b {
struct LayerParallelFlashAttentionParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0; // headDim
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama_13b";
    float qScale = 1.0;
};

atb::Status LayerParallelFlashAttentionOperation(const LayerParallelFlashAttentionParam &param,
    atb::Operation **operation);

static atb::Operation *CreateLayerParallelFlashAttentionOperation(const nlohmann::json &paramJson)
{
    LayerParallelFlashAttentionParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMA13BLayerEncoder headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank << ", rankSize:"
                  << param.rankSize;
    atb::Operation *op;
    LayerParallelFlashAttentionOperation(param, &op);
    return op;
}

class LayerPrallelFlashAttentionBinder : public HostTensorBinder {
public:
    LayerPrallelFlashAttentionBinder();
    virtual ~LayerPrallelFlashAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};

} // namespace llama_13b
} // namespace atb_speed
#endif