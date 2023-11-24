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

#ifndef ATB_SPEED_LAYERS_MLP_H
#define ATB_SPEED_LAYERS_MLP_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "common.h"

namespace atb_speed {
namespace common {


class MlpWithBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpWithBiasId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
        IN_BIAS_UP_ID,                      //
        IN_BIAS_DOWN_ID,                    //
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
    };
};

class MlpWithoutBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
        IN_BIAS_UP_ID,                      // no need
        IN_BIAS_DOWN_ID,                    // no need
    };
};

struct MlpParam {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    bool transpose = false;
    bool isBias = false;
};

atb::Status MlpLayer(const MlpParam &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed
#endif
