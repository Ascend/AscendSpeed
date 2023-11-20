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
#include "context.h"
#include <atb_speed/utils/singleton.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "buffer_device.h"
#include "buffer_host.h"

namespace atb_speed {
Context::Context()
{
    uint64_t bufferRing = GetHostTilingBufferRing();
    uint64_t bufferSize = GetHostTilingBufferSize();
    ATB_LOG(FATAL) << "Context hosttiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    hostTilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        hostTilingBuffers_.at(i).reset(new BufferHost(bufferSize));
    }

    bufferRing = GetTilingBufferRing();
    bufferSize = GetTilingBufferSize();
    ATB_LOG(FATAL) << "Context tiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    tilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        tilingBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetWorkspaceBufferRing();
    bufferSize = GetWorkspaceBufferSize();
    ATB_LOG(FATAL) << "Context workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetIntermediateBufferRing();
    bufferSize = GetIntermediateBufferSize();
    ATB_LOG(FATAL) << "Context intermediate bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    intermediateBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        intermediateBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

Context::~Context() {}

void *Context::GetHostTilingBuffer(uint64_t bufferSize)
{
    if (hostTilingBufferOffset_ == hostTilingBuffers_.size()) {
        hostTilingBufferOffset_ = 0;
    }
    return hostTilingBuffers_.at(hostTilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *Context::GetTilingBuffer(uint64_t bufferSize)
{
    if (tilingBufferOffset_ == tilingBuffers_.size()) {
        tilingBufferOffset_ = 0;
    }
    return tilingBuffers_.at(tilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *Context::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers_.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBuffers_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

void *Context::GetIntermediateBuffer(uint64_t bufferSize)
{
    if (intermediateBufferOffset_ == intermediateBuffers_.size()) {
        intermediateBufferOffset_ = 0;
    }
    return intermediateBuffers_.at(intermediateBufferOffset_++)->GetBuffer(bufferSize);
}

uint64_t Context::GetHostTilingBufferRing()
{
    const char *envStr = std::getenv("ATB_CONTEXT_HOSTTILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoll(envStr);
}

uint64_t Context::GetHostTilingBufferSize()
{
    const char *envStr = std::getenv("ATB_CONTEXT_HOSTTILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t Context::GetTilingBufferRing()
{
    const char *envStr = std::getenv("ATB_CONTEXT_TILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoll(envStr);
}

uint64_t Context::GetTilingBufferSize()
{
    const char *envStr = std::getenv("ATB_CONTEXT_TILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t Context::GetWorkspaceBufferRing()
{
    const char *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoll(envStr);
}

uint64_t Context::GetWorkspaceBufferSize()
{
    const char *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}

uint64_t Context::GetIntermediateBufferRing()
{
    const char *envStr = std::getenv("ATB_CONTEXT_INTERMEDIATE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return atoll(envStr);
}

uint64_t Context::GetIntermediateBufferSize()
{
    const char *envStr = std::getenv("ATB_CONTEXT_INTERMEDIATE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return atoll(envStr);
}
} // namespace atb_speed