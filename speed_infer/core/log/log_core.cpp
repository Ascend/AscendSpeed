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
#include "atb_speed/log/log_core.h"
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "atb_speed/log/log_sink_stdout.h"
#include "atb_speed/log/log_sink_file.h"

namespace atb_speed {
static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("ATB_LOG_TO_STDOUT");
    return envLogToStdout != nullptr && strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnv()
{
    const char* envLogToStdout = std::getenv("ATB_LOG_TO_FILE");
    return envLogToStdout != nullptr && strcmp(envLogToStdout, "1") == 0;
}

static LogLevel GetLogLevelFromEnv()
{
    const char* env = std::getenv("ATB_LOG_LEVEL");
    if (env == nullptr) {
        return LogLevel::WARN;
    }
    std::string envLogLevel(env);
    std::transform(envLogLevel.begin(), envLogLevel.end(), envLogLevel.begin(), ::toupper);
    static std::unordered_map<std::string, LogLevel> levelMap{
        {"TRACE", LogLevel::TRACE}, {"DEBUG", LogLevel::DEBUG}, {"INFO", LogLevel::INFO},
        {"WARN", LogLevel::WARN}, {"ERROR", LogLevel::ERROR}, {"FATAL", LogLevel::FATAL}
    };
    auto levelIt = levelMap.find(envLogLevel);
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::WARN;
}

LogCore::LogCore()
{
    level_ = GetLogLevelFromEnv();
    if (GetLogToStdoutFromEnv()) {
        AddSink(std::make_shared<LogSinkStdout>(level_));
    }
    if (GetLogToFileFromEnv()) {
        AddSink(std::make_shared<LogSinkFile>(level_));
    }
    levelCounts_.resize(static_cast<int>(LogLevel::FATAL) + 1);
    for (size_t i = 0; i < levelCounts_.size(); ++i) {
        levelCounts_.at(i) = 0;
    }
}

LogCore &LogCore::Instance()
{
    static LogCore logCore;
    return logCore;
}

LogLevel LogCore::GetLogLevel() const
{
    return level_;
}

void LogCore::SetLogLevel(LogLevel level)
{
    level_ = level;
}

void LogCore::Log(const LogEntity &logEntity)
{
    levelCounts_.at(static_cast<int>(logEntity.level)) += 1;
    for (auto &sink : sinks_) {
        sink->Log(logEntity);
    }
}

void LogCore::AddSink(const std::shared_ptr<LogSink> sink)
{
    sinks_.push_back(sink);
}

const std::vector<std::shared_ptr<LogSink>> &LogCore::GetAllSinks() const
{
    return sinks_;
}

atb::SVector<uint64_t> LogCore::GetLogLevelCount() const
{
    return levelCounts_;
}
}