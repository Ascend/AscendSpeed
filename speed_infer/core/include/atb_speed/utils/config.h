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
#ifndef ATB_SPEED_UTILS_CONFIG_H
#define ATB_SPEED_UTILS_CONFIG_H
#include <string>
#include <vector>
#include <set>

namespace atb_speed {
class Config {
public:
    Config();
    ~Config();
    static std::string GetSaveTensorDir();
    bool IsSaveTensor() const;
    void DisableSaveTensor();
    uint64_t GetSaveTensorMaxNum() const;
    bool IsConvertNCHWToND() const;
    bool IsSaveTensorForRunner(const std::string &runnerName) const;
    bool IsTorchTensorFormatCast() const;
    bool IsUseTilingCopyStream() const;
    bool IsLayerInternalTensorReuse() const;

private:
    static bool IsEnable(const char *env, bool enable = false);
    void InitSaveTensor();
    void InitSaveTensor(const char *env, std::set<std::string> &nameSet);

private:
    bool isSaveTensor_ = false;
    uint64_t saveTensorMaxNum_ = 1;
    bool isConvertNCHWToND_ = false;
    bool isTorchTensorFormatCast_ = true;
    bool isUseTilingCopyStream_ = false;
    std::set<std::string> saveTensorRunnerNameSet_;
    bool isLayerInternalTensorReuse_ = false;
};
} // namespace atb_speed
#endif