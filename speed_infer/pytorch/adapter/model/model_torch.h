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
#ifndef MODEL_MODEL_TORCH_H
#define MODEL_MODEL_TORCH_H
#include <string>
#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <atb_speed/utils/timer.h>
#include "atb_speed/base/model.h"

class ModelTorch : public torch::CustomClassHolder {
public:
    ModelTorch(std::string modelName);
    ~ModelTorch();
    void SetParam(std::string param);
    void SetWeight(std::vector<torch::Tensor> atWeightTensors);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors, std::string param);
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors, std::string param);
    c10::intrusive_ptr<ModelTorch> clone() const { return c10::make_intrusive<ModelTorch>(modelName_); }

private:
    void AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors);
    void ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                        const std::string &param);
    std::string GetSaveTensorDir();

private:
    std::string modelName_;
    std::shared_ptr<atb_speed::Model> model_;
    uint64_t executeCount_ = 0;
    uint64_t modelId_ = 0;
};

#endif