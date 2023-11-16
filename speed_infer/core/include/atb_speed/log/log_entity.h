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
#ifndef ATB_SPEED_LOG_LOGENTITY_H
#define ATB_SPEED_LOG_LOGENTITY_H
#include <chrono>
#include <string>

namespace atb_speed {
    enum class LogLevel {
        TRACE = 0,
        DEBUG,
        INFO,
        WARN,
        ERROR,
        FATAL
    };

    std::string LogLevelToString(LogLevel level);

    struct LogEntity {
        std::chrono::system_clock::time_point time;
        size_t processId = 0;
        size_t threadId = 0;
        LogLevel level = LogLevel::TRACE;
        const char* fileName = nullptr;
        int line = 0;
        const char *funcName = nullptr;
        std::string content;
    };
}
#endif // !ATB_SPEED_LOG_LOGENTITY_H
