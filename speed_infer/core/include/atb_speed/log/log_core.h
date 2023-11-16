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
#ifndef ATB_SPEED_LOG_LOGCORE_H
#define ATB_SPEED_LOG_LOGCORE_H
#include <memory>
#include <vector>
#include "atb_speed/log/log_entity.h"
#include "atb_speed/log/log_sink.h"
#include "atb/svector.h"

namespace atb_speed {
class LogCore {
public:
    LogCore();
    ~LogCore() = default;
    static LogCore &Instance();
    LogLevel GetLogLevel() const;
    void SetLogLevel(LogLevel level);
    void Log(const LogEntity &logEntity);
    void AddSink(const std::shared_ptr<LogSink> sink);
    const std::vector<std::shared_ptr<LogSink>> &GetAllSinks() const;
    atb::SVector<uint64_t> GetLogLevelCount() const;

private:
    std::vector<std::shared_ptr<LogSink>> sinks_;
    LogLevel level_ = LogLevel::INFO;
    atb::SVector<uint64_t> levelCounts_;
};
}
#endif // ATB_SPEED_LOG_LOGCORE_H
