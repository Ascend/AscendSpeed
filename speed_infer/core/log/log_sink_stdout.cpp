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
#include "atb_speed/log/log_sink_stdout.h"
#include <iostream>
#include <iomanip>

namespace atb_speed {
LogSinkStdout::LogSinkStdout(LogLevel level) : LogSink(level) {}
const int MICROSECOND = 1000000;
void LogSinkStdout :: LogImpl(const LogEntity &logEntity)
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity.time);
    int us = 
        std::chrono::duration_cast<std::chrono::microseconds>(logEntity.time.time_since_epoch()).count() % MICROSECOND;
    std::cout << "[" << std::put_time(std::localtime(&tmpTime), "%F %T") << "." << us << "] [" <<
        LogLevelToString(logEntity.level) << "] [" << logEntity.processId << "] [" << logEntity.threadId << "] [" <<
        logEntity.fileName << ":" << logEntity.line << "]" << logEntity.content << std::endl;       
}
} // namespace atb