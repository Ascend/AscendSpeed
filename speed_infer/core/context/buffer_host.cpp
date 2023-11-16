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
#include "buffer_host.h"
#include "atb_speed/log.h"

namespace atb_speed {
BufferHost::BufferHost(uint64_t bufferSize)
{
    ATB_LOG(INFO) << "BufferHost::BufferHost called, bufferSize:" << bufferSize;
    if (bufferSize > 0) {
        buffer_.resize(bufferSize);
    }
}

BufferHost::~BufferHost() {}

void *BufferHost::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= buffer_.size()) {
        ATB_LOG(INFO) << "BufferHost::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << buffer_.size()
                      << ", not new host mem";
        return buffer_.data();
    }

    ATB_LOG(INFO) << "BufferHost::GetBuffer success, bufferSize:" << bufferSize;
    buffer_.resize(bufferSize);
    ATB_LOG(INFO) << "BufferHost::GetBuffer success, buffer:" << buffer_.data();
    return buffer_.data();
}
} // namespace atb_speed