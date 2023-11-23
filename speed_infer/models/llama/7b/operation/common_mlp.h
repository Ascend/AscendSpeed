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
#ifndef ATB_SPEED_MODELS_LLAMA_7B_COMMON_MLP_H
#define ATB_SPEED_MODELS_LLAMA_7B_COMMON_MLP_H

#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"
#include "atb_speed/log.h"

namespace atb_speed {
namespace llama_7b {
struct CommonMlpParam {
    bool transpose = false;
};

atb::Status CommonMlp(const CommonMlpParam &param, atb::Operation **operation);

static atb::Operation *CreateCommonMlp(const nlohmann::json &paramJson)
{
    CommonMlpParam param;
    if (paramJson.contains("transpose")) {
        param.transpose = paramJson["transpose"].get<bool>();
    }
    ATB_LOG(INFO) << "CommonMlpParam";
    atb::Operation *op;
    CommonMlp(param, &op);
    return op;
}
} // namespace llama_7b
} // namespace atb_speed
#endif