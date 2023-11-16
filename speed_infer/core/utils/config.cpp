
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
#include "atb_speed/utils/config.h"
#include <string>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <atb_speed/utils/match.h>
#include <atb_speed/utils/str_split.h>
#include "atb_speed/log.h"

namespace atb_speed {
Config::Config()
{
    InitSaveTensor();
    isSaveTensor_ = IsEnable("ATB_SAVE_TENSOR");
    isConvertNCHWToND_ = IsEnable("ATB_CONVERT_NCHW_TO_ND");
    isTorchTensorFormatCast_ = IsEnable("ATB_TORCH_TENSOR_FORMAT_CAST");
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    isLayerInternalTensorReuse_ = IsEnable("ATB_LAYER_INTERNAL_TENSOR_REUSE");

    ATB_LOG(FATAL) << "Config:\nIsSaveTensor:" << isSaveTensor_ << " \nIsConvertNCHWToND:" << isConvertNCHWToND_
                   << "\nIsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                   << "\nIsLayerInternalTensorReuse:" << isLayerInternalTensorReuse_;
}

Config::~Config() {}

std::string Config::GetSaveTensorDir()
{
    std::ostringstream pid;
    pid << getpid();
    const char *dumpEnvStr = std::getenv((pid.str() + "_DUMP_PATH").c_str());
    const char *envStr = std::getenv("ATB_HOME_PATH");
    if (!envStr) {
        return "tensors/thread_" + pid.str();
    }
    if (dumpEnvStr) {
        return std::string(envStr) + "/tensors/" + std::string(dumpEnvStr);
    }
    return std::string(envStr) + "/tensors/" + "thread_" + pid.str();
}

bool Config::IsEnable(const char *env, bool enable)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsSaveTensor() const { return isSaveTensor_; }

void Config::DisableSaveTensor() { isSaveTensor_ = false; }

uint64_t Config::GetSaveTensorMaxNum() const { return saveTensorMaxNum_; }

bool Config::IsTorchTensorFormatCast() const{ return isTorchTensorFormatCast_; };

bool Config::IsConvertNCHWToND() const { return isConvertNCHWToND_; }

bool Config::IsUseTilingCopyStream() const {return isUseTilingCopyStream_;}

bool Config::IsSaveTensorForRunner(const std::string &runnerName) const
{
    if (saveTensorRunnerNameSet_.empty()) {
        return true;
    }

    for (auto &name : saveTensorRunnerNameSet_) {
        if (atb_speed::StartsWith(runnerName, name)) {
            return true;
        }
    }
    return false;
}

void Config::InitSaveTensor()
{
    InitSaveTensor("ATB_SAVE_TENSOR_RUNNER", saveTensorRunnerNameSet_);
    const char *envStr = std::getenv("ATB_SAVE_TENSOR_MAX");
    if (envStr) {
        saveTensorMaxNum_ = atoll(envStr);
    }
}

void Config::InitSaveTensor(const char *env, std::set<std::string> &nameSet)
{
    const char *envStr = std::getenv(env);
    if (!envStr) {
        return;
    }

    std::vector<std::string> names;
    atb_speed::StrSplit(std::string(envStr), ',', names);

    for (auto &name : names) {
        nameSet.insert(name);
        ATB_LOG(INFO) << env << " name:" << name;
    }
}

bool Config::IsLayerInternalTensorReuse() const
{
    return isLayerInternalTensorReuse_;
}

} // namespace atb_speed