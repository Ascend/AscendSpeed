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

#ifndef ATB_SPEED_LOG_LOGSTREAM_H
#define ATB_SPEED_LOG_LOGSTREAM_H
#include <sstream>
#include "atb_speed/log/log_entity.h"

namespace atb_speed {
class LogStream {
public:
    LogStream(const char *filePath, int line, const char *funcName, LogLevel level);
    ~LogStream();
    template <typename T> LogStream &operator << (const T &value)
    {
        stream_ << value;
        return *this;
    }
    void Format(const char *format, ...);

private:
    LogEntity logEntity_;
    std::stringstream stream_;
    bool useStream_ = true;
};
}   // namespace atb_speed
#endif
