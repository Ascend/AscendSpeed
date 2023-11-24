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
#ifndef LLAMA_7BLAYER_FUSION_OPERATION_H
#define LLAMA_7BLAYER_FUSION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
namespace llama_7b {
struct LayerFusionParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int layerId = 0;
    float qScale = 1.0;
    atb::SVector<int32_t> seqLen;
    atb::SVector<int32_t> tokenOffset;
    std::string model = "llama7b";
    int rotaryCoeff = 2;
};
atb::Status FusionLayerOperation(const LayerFusionParam &param, atb::Operation **operation);


static atb::Operation *CreateLayerFusionOperation(const nlohmann::json &paramJson)
{
    LayerFusionParam param;
    param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    param.headNum = paramJson["headNum"].get<int>();
    param.dk = paramJson["dk"].get<int>();
    param.layerId = paramJson["layerId"].get<int>();
    param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    param.qScale = paramJson["qScale"].get<float>();
    param.model = paramJson["model"].get<std::string>();
    ATB_LOG(INFO) << "LLaMA7BLayerFusionParam rmsNormEps:" << param.rmsNormEps << ", headNum:" << param.headNum
                  << ", dk:" << param.dk << ", layerid:" << param.layerId   << ", qScale:" << param.qScale
                  << ", model:"<<param.model;
    atb::Operation *op;
    atb_speed::llama_7b::FusionLayerOperation(param, &op);
    return op;
}

class LayerFusionBinder : public HostTensorBinder {
public:
    LayerFusionBinder();
    virtual ~LayerFusionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace llama_7b
} // namespace atb_speed
#endif