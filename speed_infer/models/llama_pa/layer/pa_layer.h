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
#ifndef ATB_SPEED_MODELS_LLAMA_PA_PA_LAYER_H
#define ATB_SPEED_MODELS_LLAMA_PA_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"

namespace atb_speed {
namespace llama_pa {
struct PALayerParam {
    int rank = 0;
    int rankSize = 1;
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    bool transposedWeight = false;
    bool isPrefill = false;
    std::string backend = "hccl";
    std::string model = "llama_65b";
};

enum LayerPATensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT, // weights
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_POSITIONIDS, // inputs
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_K_CACHE, // kvcache
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,

    OUT_LAYEROUT,

    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,

    INTERMIDATE_ATTENTIONOUT, // attention output
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};

static void from_json(const nlohmann::json &paramJson, PALayerParam &param)
{
    paramJson.at("rmsNormEps").get_to(param.rmsNormEps);
    paramJson.at("headNum").get_to(param.headNum);
    paramJson.at("dk").get_to(param.dk);
    if (paramJson.contains("rank")) {
        paramJson.at("rank").get_to(param.rank);
    }
    if (paramJson.contains("rankSize")) {
        paramJson.at("rankSize").get_to(param.rankSize);
    }
    if (paramJson.contains("transposedWeight")) {
        paramJson.at("transposedWeight").get_to(param.transposedWeight);
    }
    if (paramJson.contains("isPrefill")) {
        paramJson.at("isPrefill").get_to(param.isPrefill);
    }
    if (paramJson.contains("backend")) {
        paramJson.at("backend").get_to(param.backend);
    }
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation);

static atb::Operation *CreatePALayer(const nlohmann::json &paramJson)
{
    ATB_LOG(INFO) << GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    atb::Operation *op;
    PALayer(paramJson.get<PALayerParam>(), &op);
    return op;
}

class FlashAttentionHostBinder : public HostTensorBinder {
public:
    FlashAttentionHostBinder();

    virtual ~FlashAttentionHostBinder();

    void ParseParam(const nlohmann::json &paramJson) override;

    void BindTensor(atb::VariantPack &variantPack) override;

private:
    atb::SVector<int32_t> seqLen_;
};

} // namespace llama_pa
} // namespace atb_speed
#endif