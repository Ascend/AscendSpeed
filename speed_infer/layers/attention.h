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

#ifndef ATB_SPEED_ATTENTION_LAYERS_H
#define ATB_SPEED_ATTENTION_LAYERS_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"

#include "common.h"

namespace atb_speed {
namespace common {
class FlashAttentionWithROPE : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum FlashAttentionWithROPEId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize]

        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize]
        INTERMEDIATE_MIX_QKV,               // [batch, seqLen, 3*hiddenSize]
    };
};

struct FTWithROPEParam {
    bool isBias = false;
};

atb::Status FlashAttentionWithROPELayer(const FTWithROPEParam &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed
#endif
