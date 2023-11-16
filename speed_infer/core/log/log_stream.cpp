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
#include "atb_speed/log/log_stream.h"
#include <atb_speed/log.h>
#include <thread>
#include <iostream>
#include <cstring>
#include <cstdarg>
#include <securec.h>
#include <unistd.h>
#include <syscall.h>
#include "atb_speed/log/log_core.h"

namespace atb_speed {
LogStream::LogStream(const char *filePath, int line, const char *funcName, LogLevel level)
{
    const char *str = strrchr(filePath, '/');
    if (str) {
        logEntity_.fileName = str + 1;
    } else {
        logEntity_.fileName = filePath;
    }
    logEntity_.time = std::chrono::system_clock::now();
    logEntity_.level = level;
    logEntity_.processId = static_cast<uint32_t>(syscall(SYS_getpid));
    logEntity_.threadId = static_cast<uint32_t>(syscall(SYS_gettid));
    logEntity_.funcName = funcName;
    logEntity_.line = line;
}
void LogStream::Format(const char *format, ...)
{
    useStream_ = false;
    const int maxBufferLenth = 1024;
    std::string content;
    va_list args;
    va_start(args, format);
    char buffer[maxBufferLenth + 1] = {0};
    int ret = vsnprintf_s(buffer, maxBufferLenth, maxBufferLenth, format, args);
    if (ret < 0) {
        ATB_LOG(ERROR) << "vsnprintf_s ERROR! Error Code:" << ret;
        return;
    }
    va_end(args);
    content.resize(ret + 1);
    va_start(args, format);
    int ref = vsnprintf_s(&content.front(), content.size(), maxBufferLenth, format, args);
    if (ref < 0) {
        ATB_LOG(ERROR) << "vsnprintf_s ERROR! Error Code:" << ref;
        return;
    }
    va_end(args);
    logEntity_.content = content;
} 
LogStream::~LogStream()
{
    if (useStream_) {
        logEntity_.content = stream_.str();
    }
    LogCore::Instance().Log(logEntity_);
}
}