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

#ifndef ATB_SPEED_LAYERS_MLP_GATE_V2_H
#define ATB_SPEED_LAYERS_MLP_GATE_V2_H

#include <atb/atb_infer.h>
#include "nlohmann/json.hpp"
#include "atb_speed/log.h"
#include "common.h"
#include "parallel_layer_v2.h"

namespace atb_speed {
namespace common {
struct MlpGateParamV2 {
    atb::infer::ActivationType activationType;
    bool transposeB = false;
    bool isBias = false;
    bool isPack = false;
    bool isQuant = false;
    CommParam commDownParam;
    QuantParam quantUpParam;
    QuantParam quantGateParam;
    QuantParam quantDownParam;
};


atb::Status MlpGateLayerV2(const MlpGateParamV2 &param, atb::Operation **operation);


} // namespace common
} // namespace atb_speed
#endif
