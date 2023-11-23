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
#ifndef ATB_SPEED_MODELS_LLAMA_SELF_ATTENTION_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA_SELF_ATTENTION_OPERATION_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
struct llamaSelfAttentionParam {
    bool transKey = false;
    int64_t dk = 0;
    int64_t headNum = 0;
    int64_t layerId = 0;
    float preScale = 0;
    float postScale = 0;
    int64_t numHeadsPerPartition = 0;
    int64_t hiddenSizePerHead = 0;
    int64_t numGroupsPerPartition = 0;
    bool transpose = true;
    std::string model = "openbert";
};

atb::Status CreateLlamaSelfAttentionOperation(const llamaSelfAttentionParam &param, atb::Operation **operation);
} // namespace atb_speed
#endif