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
#ifndef ATB_SPEED_MODELS_LLAMA_7B_WITH_FUSION_MODEL_H
#define ATB_SPEED_MODELS_LLAMA_7B_WITH_FUSION_MODEL_H
#include "atb_speed/base/model.h"


namespace atb_speed {
namespace llama_7b {
class FusionModel : public Model {
public:
    struct Param {
        double rmsNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        float qkScale = 1.0;
        int rotaryCoeff = 2;
        atb::SVector<int32_t> tokenOffset = {};
        atb::SVector<int32_t> seqLen = {};
        void FromString(const std::string &param);
    };

    explicit FusionModel(const std::string &param);
    ~FusionModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    void BuildGraph() override;
    Param param_;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    atb::SVector<int32_t> tokenOffset_;
    atb::SVector<int32_t> seqLen_;
    int32_t layerId_ = 0;
};
} // namespace llama_7b
} // namespace atb_speed
#endif
