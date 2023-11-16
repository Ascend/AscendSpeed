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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or flagimplied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "atb_speed/base/handle.h"
#include <thread>
#include "pytorch/adapter/utils/utils.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {
thread_local atb_speed::Handle localHandle;


void InitLocalContext()
{
    uint64_t tilingBufferNumMask = 0x0000000000000070;
    uint64_t tilingBufferSizeMask = 0x0000000000000002;
    uint64_t flag = tilingBufferNumMask | tilingBufferSizeMask;
    if (atb_speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
        flag = flag | atb::MULTI_STREAM_MASK;
    }
    atb::CreateContext(&localHandle.contextPtr_, flag);
    atb::Context *contextPtr = localHandle.contextPtr_;
    contextPtr->SetExecuteStream(Utils::GetCurrentStream());
}
}