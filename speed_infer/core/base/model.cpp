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
#include "atb_speed/base/model.h"
#include <nlohmann/json.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/core/npu/register/OptionsManager.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <atb/types.h>
#include <atb_speed/utils/timer.h>
#include <atb_speed/utils/singleton.h>
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/statistic.h"
#include "atb_speed/log.h"
#include "pytorch/adapter/utils/utils.h"
#include "core/context/context.h"
#include "atb_speed/utils/singleton.h"

namespace atb_speed {

static bool IsTensorDimsEqual(const atb::Dims &left, const atb::Dims &other)
{
    if (left.dimNum != other.dimNum) {
        return false;
    }
    
    for (uint64_t i = 0; i < left.dimNum; ++i) {
        if (left.dims[i] != other.dims[i]) {
            return false;
        }
    }
    return true;
}


std::string Model::Graph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < weightTensors.size(); ++i) {
        ss << "weightTensors["<< i <<"]:" << &weightTensors.at(i) << " " << 
            TensorUtil::TensorToString(weightTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::TensorToString(inTensors.at(i)) << 
            std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " " << TensorUtil::TensorToString(outTensors.at(i)) << 
            std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " " << 
            TensorUtil::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << node.operation.get() << ", operationName:" << 
        node.operation->GetName() << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt) << 
                std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt) << 
                std::endl;
        }
    }
    return ss.str();
}

void Model::Graph::Init()
{
    for (size_t i = 0; i < nodes.size(); i++) {
        auto &node = nodes.at(i);
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
        node.torchTensors.resize(node.outTensors.size());
    }
    InitTensorType();
}

void Model::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) =
                IsInternalTensor(node.inTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) =
                IsInternalTensor(node.outTensors.at(i)) ? Model::INTERMEDIATE_TENSOR : Model::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Model::Graph::IsInternalTensor(const atb::Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

Model::Model(const std::string &modelName, const std::string &param) : modelName_(modelName), param_(param)
{
    aclrtGetDevice(&currentDevId_);

    const char *envStr = std::getenv("TASK_QUEUE_ENABLE");
    isTaskQueueEnable_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;

    envStr = std::getenv("ATB_OPERATION_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr != nullptr && std::string(envStr) == "1") ? true : false;
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        std::thread thread = std::thread(std::bind(&Model::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ATB_LOG(FATAL) << modelName_ << " new, isTaskQueueEnable:" << isTaskQueueEnable_ << ", isUsePlanExecuteAsync:" << 
        isUsePlanExecuteAsync_ << ", currentDevId:" << currentDevId_;
}

Model::~Model() {}

void Model::Init()
{
    BuildGraph();
    graph_.Init();
    ATB_LOG(INFO) << modelName_ << " init graph:\n" << graph_.ToString();
}

void Model::SetWeight(const std::vector<atb::Tensor> &weightTensors)
{
    if (graph_.weightTensors.size() != weightTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " weightTensors.size:" << weightTensors.size() << " != "
                       << " graph.weightTensors.size:" << graph_.weightTensors.size();
        return;
    }

    graph_.weightTensors = weightTensors;
}

atb::Status Model::Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
                           std::vector<atb::Tensor> &outTensors, const std::string &param)
{
    if (graph_.inTensors.size() != inTensors.size() || graph_.outTensors.size() != outTensors.size()) {
        ATB_LOG(ERROR) << modelName_ << " graph.inTensors.size:" << graph_.inTensors.size() << ", inTensors.size:" <<
            inTensors.size() << ", graph.outTensors.size:" << graph_.outTensors.size() << ", outTensors.size:" << 
            outTensors.size();
        return atb::ERROR_INVALID_GRAPH;
    }

    ParseParam(param);

    timer_.Reset();
    allTaskFinish_ = false;
    context_ = context;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    ATB_LOG(INFO) << modelName_ << " execute start, executeCount:" << executeCount_ << ", graph:\n"
                  << graph_.ToString();

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        BuildNodeVariantPack(nodeId);
        BindParamHostTensor(nodeId);
        atb::Status st = ExecuteNode(nodeId);
        if (st != 0) {
            return st;
        }
    }

    WaitAsyncPlanExecuteFinish();

    GetSingleton<Statistic>().totalTime += timer_.ElapsedMicroSecond();
    ATB_LOG(FATAL) << modelName_ << " executeCount:" << executeCount_ << ", Statistic:[" << 
        GetSingleton<Statistic>().ToString() << "]";
    GetSingleton<Statistic>().Reset();

    executeCount_++;

    return atb::NO_ERROR;
}

atb::Status Model::ParseParam(const std::string &param)
{
    return atb::NO_ERROR;
}

atb::Status Model::BindParamHostTensor(uint32_t nodeId)
{
    return atb::NO_ERROR;
}

void Model::BuildNodeVariantPack(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);

    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        inTensorDescs.at(i) = node.inTensors.at(i)->desc;
        ATB_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] inTensors[" << i << "]:" << 
            TensorUtil::TensorToString(node.variantPack.inTensors.at(i));
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(node.operation->GetOutputNum());
    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);

    ATB_LOG_IF(st != 0, FATAL) << modelName_ << " nodes[" << nodeId << "] "
                               << " infer shape fail, error code: " << st;
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] outTensorDescs[" << i << "]:" << 
            TensorUtil::TensorDescToString(outTensorDescs.at(i));
    }

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::INTERMEDIATE_TENSOR) {
            if ((uint64_t)node.torchTensors.at(i).numel() != TensorUtil::GetTensorNumel(outTensorDescs.at(i))) {
                BuildInternalTensor(outTensorDescs.at(i), nodeId, i);
            }
            node.variantPack.outTensors.at(i) = Utils::AtTensor2Tensor(node.torchTensors.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
        if (!TensorUtil::TensorDescEqual(node.variantPack.outTensors.at(i).desc, outTensorDescs.at(i))) {
            ATB_LOG(FATAL) << modelName_ << "  nodes[" << nodeId << "] new outTensorDescs[" << i << "]:" << 
                TensorUtil::TensorDescToString(outTensorDescs.at(i)) << ", node.variantPack.outTensors.at[" << i <<
                     "].desc:" << TensorUtil::TensorDescToString(node.variantPack.outTensors.at(i).desc);
        }
    }
}

void Model::BuildInternalTensor(const atb::TensorDesc &tensorDesc, int nodeId, size_t tensorId)
{
    auto &node  = graph_.nodes.at(nodeId);
    ATB_LOG(INFO) << modelName_ << " nodes[" << nodeId << "] new outtensors[" << tensorId << "]";

    if (GetSingleton<Config>().IsLayerInternalTensorReuse()) {
        torch::Tensor preTensor = FindPreInternalTensor(tensorDesc, nodeId, tensorId);
        if (preTensor.numel() > 0) {
            node.torchTensors.at(tensorId) = preTensor;
            return;
        }
    }

    Timer timer;
    node.torchTensors.at(tensorId) = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
    GetSingleton<Statistic>().createTensorTime += timer.ElapsedMicroSecond();
}

atb::Status Model::ExecuteNode(int nodeId)
{
    ExecuteNodeView(nodeId);
    auto &node = graph_.nodes.at(nodeId);

    Timer timer;
    atb::Status st = node.operation->Setup(node.variantPack, node.workspaceSize);
    GetSingleton<Statistic>().planSetupTime += timer.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << modelName_ << " setup node[" << nodeId << "] fail, not call execute";
        return st;
    }

    ATB_LOG(INFO) << modelName_ << " get node[" << nodeId << "] workspace size:" << node.workspaceSize;

    if (node.workspaceSize > 0) {
        node.workspace = GetSingleton<Context>().GetWorkspaceBuffer(node.workspaceSize);
    }

    if (isUsePlanExecuteAsync_) {
        Timer timer;
        ExecutePlanAsync(nodeId);
        GetSingleton<Statistic>().planAsyncTime += timer.ElapsedMicroSecond();
    } else {
        st = ExecutePlanSync(nodeId);   
    }
    return st;
}

void Model::ThreadProcessTask()
{
    ATB_LOG(FATAL) << modelName_ << " thread process operations start";
    int ret = aclrtSetDevice(currentDevId_);
    ATB_LOG_IF(ret != 0, ERROR) << "AsdRtDeviceSetCurrent fail, error:" << ret;

    size_t processTaskCount = 0;
    while (true) {
        int nodeId = PopTask();
        atb::Status st = ExecutePlanSync(nodeId);
        if (st != 0) {
            allTaskFinish_ = true;
            processTaskCount = 0;
            return;
        }
        processTaskCount++;
        if (processTaskCount == graph_.nodes.size()) {
            ATB_LOG(INFO) << modelName_ << " thread process all operations";
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

atb::Status Model::ExecutePlanSync(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    atb::VariantPack &variantPack = node.variantPack;

    ATB_LOG(INFO) << modelName_ << "execute node[" << nodeId << "] start";
    Timer timer;
    atb::Status st = node.operation->Execute(variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    GetSingleton<Statistic>().planExecuteTime += timer.ElapsedMicroSecond();
    if (st != 0) {
        ATB_LOG(ERROR) << "execute node[" << nodeId << "] fail, error code: " << st;
    }
    return st;
}

void Model::ExecutePlanAsync(int nodeId)
{
    if (isTaskQueueEnable_) {
#ifdef TORCH_SETCUSTOMHANDLER
        at_npu::native::OpCommand cmd;
        cmd.Name(modelName_ + std::to_string(nodeId));
        cmd.SetCustomHandler([=]() {
            ExecutePlanSync(nodeId);
            return 0;
        });
        cmd.Run();
#else
        ATB_LOG(FATAL) << modelName_ << " torch_npu is low, can't support SetCustomHandler";
#endif
    } else {
        PushTask(nodeId);
    }
}

void Model::PushTask(int nodeId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(nodeId);
    lock.unlock();
    cond_.notify_one();
}

int Model::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int nodeId = taskQueue_.front();
    taskQueue_.pop();
    return nodeId;
}

void Model::WaitAsyncPlanExecuteFinish()
{
    if (isUsePlanExecuteAsync_ && !isTaskQueueEnable_) {
        while (true) {
            if (allTaskFinish_) {
                ATB_LOG(INFO) << modelName_ << " allTaskFinish is true, break";
                break;
            }
        }
    }
}

std::string Model::GetSaveTensorDir()
{
    std::string dir = std::to_string(executeCount_) + "/0_Model";
    return Config::GetSaveTensorDir() + "/" + dir;
}

void Model::ExecuteNodeView(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    if (node.inTensorReshapeFuncs.size() > 0) {
        for (size_t i = 0; i < node.inTensors.size() && node.inTensorReshapeFuncs.at(i) != nullptr; i++) {
            node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape, node.inTensors.at(i)->desc.shape);
        }
    }
}

torch::Tensor Model::FindPreInternalTensor(const atb::TensorDesc &tensorDesc, uint32_t nodeId, uint32_t tensorId) const
{
    torch::Tensor preSameTensor;
    if (nodeId == 0) {
        return preSameTensor;
    }

    for (int64_t preNodeId = nodeId - 1; preNodeId >= 0; preNodeId--) {
        auto &preNode = graph_.nodes.at(preNodeId);
        if (tensorId < preNode.torchTensors.size() && 
            IsTensorDescEqual(tensorDesc, preNode.torchTensors.at(tensorId))) {
                preSameTensor = preNode.torchTensors.at(tensorId);
                ATB_LOG(INFO) << modelName_ << " find same tensor, node[ " << nodeId << "] preNodeId:" << preNodeId <<
                ", tensorId:" << tensorId;
                return preSameTensor;
            }
    }
    ATB_LOG(INFO) << modelName_ << " not find same tensor, node[ " << nodeId << "] tensorId:" << tensorId;
    return preSameTensor;
}

bool Model::IsTensorDescEqual(const atb::TensorDesc &tensorDesc, const torch::Tensor &atTensor ) const
{
    atb::Tensor atbTensor = Utils::AtTensor2Tensor(atTensor);
    return atbTensor.desc.dtype == tensorDesc.dtype && atbTensor.desc.format == tensorDesc.format && 
        IsTensorDimsEqual(atbTensor.desc.shape, tensorDesc.shape);
}
} // namespace atb_speed