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
#ifndef ASCENDSPEED_TE_OPS_CSRC_COMMON_H
#define ASCENDSPEED_TE_OPS_CSRC_COMMON_H

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch/script.h>
#include <torch/custom_class.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, std::string input_layout,
    const c10::optional<at::Tensor> &pse_opt, const c10::optional<at::Tensor> &padding_mask_opt,
    const c10::optional<at::Tensor> &atten_mask_opt,
    double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    c10::optional<at::IntArrayRef> prefix_opt, int64_t sparse_mode, bool gen_mask_parallel, bool sync);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    std::string input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    double scale_value,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    c10::optional<at::IntArrayRef> prefix,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_add_layer_norm(
    const at::Tensor &x0,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &residual_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &rowscale_opt,
    const c10::optional<at::Tensor> &layerscale_opt,
    double p,
    double eps,
    bool prenorm,
    bool residual_in_fp32,
    bool is_rms_norm,
    bool return_dropout_mask);

#endif
