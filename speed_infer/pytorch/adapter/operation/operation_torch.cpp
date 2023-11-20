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
#include "operation_torch.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>
#include <acl/acl.h>
#include <atb_speed/utils/timer.h>
#include <atb_speed/utils/singleton.h>
#include <atb_speed/base/handle.h>
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/log.h"
#include "pytorch/adapter/utils/utils.h"
#include "core/context/context.h"
#include "operation_creator.h"

uint64_t GetNewOpId()
{
    static uint64_t opId = 0;
    uint64_t newOpId = opId++;
    return newOpId;
}

OperationTorch::OperationTorch(std::string opName) : opName_(opName), name_(opName)
{
    opId_ = GetNewOpId();
    nodeId_ = std::to_string(opId_);
    ATB_LOG(INFO) << "OperationTorch::OperationTorch, TASK_QUEUE_ENABLE:" 
    << c10_npu::option::OptionsManager().CheckQueueEnable() << ", opName:" << opName 
    << ", opId:" << opId_;
    atb_speed::InitLocalContext();
}

OperationTorch::~OperationTorch() {}

void OperationTorch::SetName(std::string name) { name_ = name; }

void OperationTorch::SetParam(std::string param)
{
    ATB_LOG(INFO) << name_ << " set param start, param:" << param;
    param_ = param;

    atb::Operation *operation = CreateOperation(opName_, param_);
    if (operation == nullptr) {
        ATB_LOG(FATAL) << name_ << " create operation fail, opName:" << opName_ << ", param:" << param_;
        return;
    }

    operation_.reset(operation);

    ATB_LOG(INFO) << name_ << " set param end";
}

std::vector<torch::Tensor> OperationTorch::ExecuteWithParam(std::vector<torch::Tensor> atInTensors,
    std::string varaintPackParam)
{
    ATB_LOG(INFO) << name_ << " execute start";
    if (!operation_) {
        SetParam(varaintPackParam);
    }

    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute fail, operation is null";
    }
    Utils::ContiguousAtTensor(atInTensors);

    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    Utils::ContiguousAtTensor(atOutTensors);

    ExecuteOutImpl(atInTensors, atOutTensors, varaintPackParam);
    return atOutTensors;
}

void OperationTorch::ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors,
    std::vector<torch::Tensor> atOutTensors, std::string varaintPackParam)
{
    ATB_LOG(INFO) << name_ << " execute out start";
    if (!operation_) {
        SetParam(varaintPackParam);
    }

    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute out fail, operation is null";
    }

    Utils::ContiguousAtTensor(atInTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors, varaintPackParam); 
}

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    ATB_LOG(INFO) << name_ << " execute start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute fail, operation is null";
    }
    Utils::ContiguousAtTensor(atInTensors);
    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
    return atOutTensors;
}

void OperationTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    ATB_LOG(INFO) << name_ << " execute out start";
    if (!operation_) {
        ATB_LOG(FATAL) << name_ << " execute out fail, operation is null";
    }

    Utils::ContiguousAtTensor(atInTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void OperationTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
    const std::string &varaintPackParam)
{
    ATB_LOG(INFO) << name_ << " execute impl execCount:" << executeCount_;
    atb_speed::Timer timer;

    if (hostTensorBinder_) {
        nlohmann::json paramJson = nlohmann::json::parse(varaintPackParam);
        hostTensorBinder_->ParseParam(paramJson);
    }

    atb::VariantPack variantPack;
    BuildVariantPack(atInTensors, atOutTensors, variantPack);

    if (hostTensorBinder_) {
        hostTensorBinder_->BindTensor(variantPack);
    }

    uint64_t workspaceSize = 0;

    atb_speed::Timer timer1;
    atb::Status st = operation_->Setup(variantPack, workspaceSize);
    atb_speed::GetSingleton<atb_speed::Statistic>().planSetupTime += timer1.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << name_ << " setup fail, not call execute, error code: " << st;
        return;
    }

    ATB_LOG(INFO) << name_ << " get plan workspace size:" << workspaceSize;

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = atb_speed::GetSingleton<atb_speed::Context>().GetWorkspaceBuffer(workspaceSize);
    }

    atb_speed::Timer timer2;
    atb::Context *contextPtr = atb_speed::localHandle.contextPtr_;
    st = operation_->Execute(variantPack, (uint8_t*)workspace, workspaceSize, contextPtr);
    atb_speed::GetSingleton<atb_speed::Statistic>().planExecuteTime += timer2.ElapsedMicroSecond();
    ATB_LOG_IF(st != 0, ERROR) << name_ << " execute plan fail, error code: " << st;
    atb_speed::GetSingleton<atb_speed::Statistic>().totalTime += timer.ElapsedMicroSecond();
    ATB_LOG(FATAL) << name_ << " executeCount:" << executeCount_ << ", statistic:[" 
    << atb_speed::GetSingleton<atb_speed::Statistic>().ToString() << "]";
    atb_speed::GetSingleton<atb_speed::Statistic>().Reset();

    executeCount_++;
}

void OperationTorch::CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors,
    std::vector<torch::Tensor> &atOutTensors)
{
    atb::SVector<atb::TensorDesc> outTensorDescs;
    ATB_LOG(INFO) << "11111";
    ATB_LOG(INFO) << operation_->GetOutputNum();
    outTensorDescs.resize(operation_->GetOutputNum());
    ATB_LOG(INFO) <<"2222222";
    atb::SVector<atb::TensorDesc> inTensorDescs;
    ATB_LOG(INFO) <<"3333333";
    for (size_t i = 0; i < atInTensors.size(); ++i) {
    auto &atInTensor = atInTensors.at(i);
        ATB_LOG(INFO) <<"44444444";
    atb::Tensor inTensor = Utils::AtTensor2Tensor(atInTensor);
        ATB_LOG(INFO) <<"555555555";
    inTensorDescs.push_back(inTensor.desc);
        ATB_LOG(INFO) << name_ << " infer shape inTensors[" << i 
        << "]:" << atb_speed::TensorUtil::TensorToString(inTensor);
    }
    atb::Status st = operation_->InferShape(inTensorDescs, outTensorDescs);
    ATB_LOG_IF(st != 0, FATAL) << name_ << " infer shape fail, error code: " << st;

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << name_ << " infer shape outTensorDescs[" << i 
        << "]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i));
        atb_speed::Timer timer;
        at::Tensor newTensor = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atb_speed::GetSingleton<atb_speed::Statistic>().createTensorTime += timer.ElapsedMicroSecond();
        atOutTensors.at(i) = newTensor;
    }
}

void OperationTorch::BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
    atb::VariantPack &variantPack)
{
    variantPack.inTensors.resize(atInTensors.size());
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ATB_LOG(INFO) << name_ << " execute start, atInTensors[" << i << "].options:" << atInTensors.at(i).options() 
        << ", data:" << atInTensors.at(i).data_ptr() 
        << ", storage_offset:" << atInTensors.at(i).storage_offset() 
        << ", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsTorchTensorFormatCast()) {
            atInTensors.at(i) = Utils::NpuFormatCast(atInTensors.at(i));
        }
        variantPack.inTensors.at(i) = Utils::AtTensor2Tensor(atInTensors.at(i));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND() &&
            variantPack.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.inTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }

    variantPack.outTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_LOG(INFO) << name_ << " execute start, atOutTensors[" << i << "].options:" << atOutTensors.at(i).options() 
        << ", data:" << atOutTensors.at(i).data_ptr() 
        << ", storage_offset:" << atOutTensors.at(i).storage_offset() 
        << ", format:" << Utils::GetTensorNpuFormat(atOutTensors.at(i));
        variantPack.outTensors.at(i) = Utils::AtTensor2Tensor(atOutTensors.at(i));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND() &&
            variantPack.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.outTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }
}

std::string OperationTorch::GetSaveTensorDir()
{
    const char *envStr = std::getenv("AIT_CMP_TASK_ID");
    std::string dir = envStr ? std::string(envStr) : std::to_string(executeCount_);
    return atb_speed::Config::GetSaveTensorDir() + "/" + dir + "/" + std::to_string(opId_) + "_OperationTorch";
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_name", &OperationTorch::SetName)
        .def("set_param", &OperationTorch::SetParam)
        .def("execute", &OperationTorch::Execute)
        .def("execute_out", &OperationTorch::ExecuteOut)
        .def("execute_with_param", &OperationTorch::ExecuteWithParam)
        .def("execute_out_with_param", &OperationTorch::ExecuteOutWithParam);
    ;
}