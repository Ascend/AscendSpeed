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
#ifndef ATB_SPEED_MODELS_LLAMALAYER_FUSION_PARALLEL_OPERATION_H
#define ATB_SPEED_MODELS_LLAMALAYER_FUSION_PARALLEL_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
struct LlamaLayerFusionParallelParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "llama13b";
    int layerId = 0;
    float qkScale = 1;
    int rotaryCoeff = 2;
    bool transpose = true;
    atb::SVector<int32_t> tokenOffset;
    atb::SVector<int32_t> seqLen;
};

atb::Status LlamaLayerFusionParallelOperation(const LlamaLayerFusionParallelParam &param, atb::Operation **operation);

class LlamaLayerFusionParallelBinder : public HostTensorBinder {
public:
    LlamaLayerFusionParallelBinder();

    virtual ~LlamaLayerFusionParallelBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int64_t> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace atb_speed
#endif