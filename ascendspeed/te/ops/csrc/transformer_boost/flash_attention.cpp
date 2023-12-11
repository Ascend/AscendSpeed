// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include <torch_npu/csrc/core/npu/SecondaryStreamGuard.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "../ops_interface.h"
#include "inc/adapter.h"

using namespace std;

const static int N = 32;
static atb::Context* msContext = nullptr;

atb::Context* GetContext()
{
    if (msContext == nullptr) {
        auto status = atb::CreateContext(&msContext);
        TORCH_CHECK(status == 0, "create context failed!");
        int32_t devId = 0;
        aclrtGetDevice(&devId);
        aclrtStream stream = c10_npu::getCurrentNPUStream(devId).stream(false);
        TORCH_CHECK(stream != nullptr, "get current stream failed");
        msContext->SetExecuteStream(stream);
    }
    return msContext;
}

std::tuple<at::Tensor, at::Tensor> fa(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                      const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                      const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num,
                                      int64_t io_layout, float keep_prob, int64_t pre_tokens, int64_t next_tokens,
                                      int64_t precise_mode, int64_t groups)
{
    atb::train::FlashAttentionParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    std::vector<at::Tensor> outTensors;
    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    std::vector<atb::Tensor> inTensors;
    auto atb_query = Input(query);
    auto atb_key = Input(key);
    auto atb_value = Input(value);
    auto atb_atten_mask = Input(atten_mask);
    auto atb_alibi_mask = Input(alibi_mask);
    auto atb_drop_mask = Input(drop_mask);

    inTensors.push_back(atb_query);
    inTensors.push_back(atb_key);
    inTensors.push_back(atb_value);
    inTensors.push_back(atb_atten_mask);
    inTensors.push_back(atb_alibi_mask);
    inTensors.push_back(atb_drop_mask);

    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        atb::Tensor inTensor = inTensors.at(i);
        if (inTensor.desc.format == ACL_FORMAT_NCHW) {
            inTensor.desc.format = ACL_FORMAT_ND;
        }
        inTensorDescs.push_back(inTensor.desc);
    }
    atb::Status status = op->InferShape(inTensorDescs, outTensorDescs);
    TORCH_CHECK(status == 0, "infershape failed!");

    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        outTensors.push_back(newTensor);
    }

    atb::Tensor atbOutTensor1 = AtTensor2Tensor(outTensors[0]);
    atb::Tensor atbOutTensor2 = AtTensor2Tensor(outTensors[1]);
    atb::VariantPack variantPack;
    variantPack.inTensors.push_back(atb_query);
    variantPack.inTensors.push_back(atb_key);
    variantPack.inTensors.push_back(atb_value);
    variantPack.inTensors.push_back(atb_atten_mask);
    variantPack.inTensors.push_back(atb_alibi_mask);
    variantPack.inTensors.push_back(atb_drop_mask);
    variantPack.outTensors.push_back(atbOutTensor1);
    variantPack.outTensors.push_back(atbOutTensor2);

    uint64_t workspaceSize = 0;
    atb::Status st = op->Setup(variantPack, workspaceSize);
    TORCH_CHECK(st == 0, "setup failed!");
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));
        workspacePtr = workspaceTensor.storage().data();
    }
    auto contextPtr = GetContext();
    auto acl_call = [op, contextPtr, variantPack, workspacePtr, workspaceSize]() -> int {
        auto st = op->Execute(variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        DestroyOperation(op);
        return 0;
    };

    at_npu::native::OpCommand cmd;
    cmd.Name("fa_forward");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    
    return std::make_tuple(outTensors[0], outTensors[1]);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fag(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out,
                                                   const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                                   const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                                   const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num, int64_t io_layout,
                                                   float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups)
{
    atb::train::FlashAttentionBackwardParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionBackwardParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    std::vector<at::Tensor> outTensors;
    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "get op failed!");

    std::vector<atb::Tensor> inTensors;
    auto atb_dy = Input(dy);
    auto atb_softmax_log_max_sum = Input(softmax_log_max_sum);
    auto atb_attention_out = Input(attention_out);
    auto atb_query = Input(query);
    auto atb_key = Input(key);
    auto atb_value = Input(value);
    auto atb_atten_mask = Input(atten_mask);
    auto atb_alibi_mask = Input(alibi_mask);
    auto atb_drop_mask = Input(drop_mask);

    inTensors.push_back(atb_dy);
    inTensors.push_back(atb_softmax_log_max_sum);
    inTensors.push_back(atb_attention_out);
    inTensors.push_back(atb_query);
    inTensors.push_back(atb_key);
    inTensors.push_back(atb_value);
    inTensors.push_back(atb_atten_mask);
    inTensors.push_back(atb_alibi_mask);
    inTensors.push_back(atb_drop_mask);

    atb::SVector<atb::TensorDesc> inTensorDescs;
    atb::SVector<atb::TensorDesc> outTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        atb::Tensor inTensor = inTensors.at(i);
        if (inTensor.desc.format == ACL_FORMAT_NCHW) {
            inTensor.desc.format = ACL_FORMAT_ND;
        }
        inTensorDescs.push_back(inTensor.desc);
    }
    atb::Status status = op->InferShape(inTensorDescs, outTensorDescs);
    TORCH_CHECK(status == 0, "infershape failed!");

    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        at::Tensor newTensor = CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        outTensors.push_back(newTensor);
    }

    atb::Tensor atbOutTensor1 = AtTensor2Tensor(outTensors[0]);
    atb::Tensor atbOutTensor2 = AtTensor2Tensor(outTensors[1]);
    atb::Tensor atbOutTensor3 = AtTensor2Tensor(outTensors[2]);
    atb::VariantPack variantPack;

    variantPack.inTensors.push_back(atb_dy);
    variantPack.inTensors.push_back(atb_softmax_log_max_sum);
    variantPack.inTensors.push_back(atb_attention_out);
    variantPack.inTensors.push_back(atb_query);
    variantPack.inTensors.push_back(atb_key);
    variantPack.inTensors.push_back(atb_value);
    variantPack.inTensors.push_back(atb_atten_mask);
    variantPack.inTensors.push_back(atb_alibi_mask);
    variantPack.inTensors.push_back(atb_drop_mask);
    variantPack.outTensors.push_back(atbOutTensor1);
    variantPack.outTensors.push_back(atbOutTensor2);
    variantPack.outTensors.push_back(atbOutTensor3);
 
    uint64_t workspaceSize = 0;
    atb::Status st = op->Setup(variantPack, workspaceSize);
    TORCH_CHECK(st == 0, "setup failed!");
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        auto workspaceTensor = at::empty({workspaceSize}, options.dtype(at::kByte));
        workspacePtr = workspaceTensor.storage().data();
    }

    auto contextPtr = GetContext();

    auto acl_call = [op, contextPtr, variantPack, workspacePtr, workspaceSize]() -> int {
        auto st = op->Execute(variantPack, (uint8_t *)workspacePtr, workspaceSize, contextPtr);
        DestroyOperation(op);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("fa_backward");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    return std::make_tuple(outTensors[0], outTensors[1], outTensors[2]);
}

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

DropOutStatus get_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    c10::TensorOptions options = self.options();
    at::Tensor mask = at::empty(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
    at::SmallVector<int64_t, N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(keep_prob, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(seed, at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong, at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

at::Tensor gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync)
{
    at::Tensor mask;

    if (gen_mask_parallel) {
        auto original_stream = c10_npu::getCurrentNPUStream();
        {
            // During the life cycle of this raii instance, the calcu stream is set as the
            // secondary stream, and tasks are distributed to the secondary stream. At the
            // same time, according to the one-stream-one-pool principle, memory is also
            // alloced from the pool of the secondary stream.
            c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
            mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
        }
    } else {
        mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
    }
    return mask;
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t> gen_mask(const at::Tensor &self, double keep_prob,
    int64_t head_num, std::string input_layout, bool gen_mask_parallel, bool sync)
{
    int64_t seed;
    int64_t offset;
    int64_t numels;
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = self.size(0) * self.size(1) * self.size(2) * self.size(2); // [B,N,S,S]
    }
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed),
            offset, numels, gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }

    return std::make_tuple(drop_mask, seed, offset, numels);
}

at::Tensor exist_gen_mask(const at::Tensor &self, double keep_prob, bool gen_mask_parallel, bool sync,
    int64_t seed, int64_t offset, int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                      gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }
    return drop_mask;
}
