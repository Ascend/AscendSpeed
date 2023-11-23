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

#include "atb_speed/base/handle.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/utils/tensor_util.h"
#include "pytorch/adapter/utils/utils.h"
#include <acl/acl.h>
#include <atb_speed/utils/singleton.h>
#include <atb_speed/utils/timer.h>
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/timer.h"
#include "model_torch.h"

#include "llama/7b/model/decoder_without_fusion_model.h"
#include "llama/7b/model/encoder_without_fusion_model.h"
#include "llama/7b/model/flashattention_model.h"
#include "llama/7b/model/fusion_model.h"
#include "llama/7b/model/rope_model.h"
#include "llama/7b/model/quant_flashattention_model.h"

uint64_t GetNewModelId()
{
    static uint64_t modelId = 0;
    uint64_t newModelId = modelId++;
    return newModelId;
}

ModelTorch::ModelTorch(std::string modelName) : modelName_(modelName)
{
    modelId_ = GetNewModelId();
    ATB_LOG(INFO) << "ModelTorch new modelName:" << modelName_ << ", modelId:" << modelId_;
    atb_speed::InitLocalContext();
}

ModelTorch::~ModelTorch() = default;

void ModelTorch::SetParam(std::string param)
{
    ATB_LOG(INFO) << "ModelTorch set param start, modelName:" << modelName_ << ", param:" << param;
    if (modelName_ == "llama_7b_decoder_without_fusion_model") {
        model_ = std::make_shared<atb_speed::llama_7b::DecoderWithoutFusionModel>(param);
    } else if (modelName_ == "llama_7b_encoder_without_fusion_model") {
        model_ = std::make_shared<atb_speed::llama_7b::EncoderWithoutFusionModel>(param);
    } else if (modelName_ == "llama_7b_decoder_rope_model") {
        model_ = std::make_shared<atb_speed::llama_7b::DecoderRopeModel>(param);
    } else if (modelName_ == "llama_7b_encoder_rope_model") {
        model_ = std::make_shared<atb_speed::llama_7b::EncoderRopeModel>(param);
    } else if (modelName_ == "llama_7b_fusion_model") {
        model_ = std::make_shared<atb_speed::llama_7b::FusionModel>(param);
    } else if (modelName_ == "llama_quant_flashattention_model") {
        model_ = std::make_shared<atb_speed::llama_7b::QuantFlashAttentionModel>(param);
    } else if (modelName_ == "llama_7b_flashattention_model") {
        model_ = std::make_shared<atb_speed::llama_7b::FlashAttentionModel>(param);
    } else {
    ATB_LOG(FATAL) << "not support modelName:" << modelName_;
        return;
    }

    model_->Init();

    ATB_LOG(INFO) << "ModelTorch set param end";
}

void ModelTorch::SetWeight(std::vector<torch::Tensor> atWeightTensors)
{
    ATB_LOG(INFO) << "ModelTorch set weight:" << atWeightTensors.size();
    for (size_t i = 0; i < atWeightTensors.size(); ++i) {
        const torch::Tensor &atTensor = atWeightTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atWeightTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
    }
    std::vector<atb::Tensor> weigthTensors;
    AtTensor2Tensor(atWeightTensors, weigthTensors);
    model_->SetWeight(weigthTensors);
}

std::vector<torch::Tensor> ModelTorch::Execute(std::vector<torch::Tensor> atInTensors, std::string param)
{
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        const torch::Tensor &atTensor = atInTensors.at(i);
        ATB_LOG(INFO) << "ModelTorch atInTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options();
    }

    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (size_t i = 0; i < inTensors.size(); ++i) {
            if (inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
                inTensors.at(i).desc.format = ACL_FORMAT_ND;
            }
        }
    }
    std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensorDescs.at(i) = inTensors.at(i).desc;
    }
    std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
    atb::Status st = model_->InferShape(inTensorDescs, outTensorDescs);
    ATB_LOG_IF(st != 0, FATAL) << "ModelTorch infer shape fail, error code: " << st;

    std::vector<torch::Tensor> atOutTensors(outTensorDescs.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_LOG(INFO) << "ModelTorch outTensorDescs[" << i
                      << "]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i));
        atb_speed::Timer timer;
        atOutTensors.at(i) = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atb_speed::GetSingleton<atb_speed::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
    }

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (size_t i = 0; i < outTensors.size(); ++i) {
            if (outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
                outTensors.at(i).desc.format = ACL_FORMAT_ND;
            }
        }
    }

    ExecuteOutImpl(inTensors, outTensors, param);

    return atOutTensors;
}

void ModelTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                            std::string param)
{
    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);

    ExecuteOutImpl(inTensors, outTensors, param);
}

void ModelTorch::ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                                const std::string &param)
{
    atb::Context *contextPtr = atb_speed::localHandle.contextPtr_;
    model_->Execute(contextPtr, inTensors, outTensors, param);
    executeCount_++;
}

void ModelTorch::AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors)
{
    for (auto &atTensor : atTensors) {
        Utils::ContiguousAtTensor(atTensor);
        atb::Tensor tensor = Utils::AtTensor2Tensor(atTensor);
        opsTensors.push_back(tensor);
    }
}

std::string ModelTorch::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/" + std::to_string(modelId_) + "_ModelTorch";
    return atb_speed::Config::GetSaveTensorDir() + "/" + dir;
}

TORCH_LIBRARY(ModelTorch, m)
{
    m.class_<ModelTorch>("ModelTorch")
        .def(torch::init<std::string>())
        .def("set_param", &ModelTorch::SetParam)
        .def("set_weight", &ModelTorch::SetWeight)
        .def("execute", &ModelTorch::Execute)
        .def("execute_out", &ModelTorch::ExecuteOut);
}