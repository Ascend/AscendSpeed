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
#include "pa_layer.h"

#include "layers/mlp_gate.h"
#include "layers/parallel_layer.h"

namespace atb_speed {
namespace llama_pa {

static const uint64_t IN_TENSOR_COUNT = 16;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 12;
static const uint64_t NODE_COUNT = 11;

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3; // dimNum: 3
    newShape.dims[0] = oldShape.dims[0]; // 0 dim: n tokens
    newShape.dims[1] = headNum;  // 1 dim: head num
    newShape.dims[2] = oldShape.dims[1] / headNum;  // 1 dim: head size
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_LOG(INFO) << __func__ << " called, headNum: " << param.headNum;
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &qkvLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitQkvNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // norm [n_tokens, hidden_size]
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    // qkv  [n_tokens, hidden_size] to [n_tokens, 3 * hidden_size]
    atb::infer::LinearParam linearParam;
    linearParam.transposeB = param.transposedWeight;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT};
    qkvLinearNode.outTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};

    // q/k/v [n_tokens, hidden_size]
    atb::infer::SplitParam splitParam = {-1, 3};
    CreateOperation(splitParam, &splitQkvNode.operation);
    splitQkvNode.inTensorIds = {INTERMIDATE_QKVMIXEDLINEAROUT};
    splitQkvNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};
    splitQkvNode.inTensorReshapeFuncs.resize(splitQkvNode.inTensorIds.size());

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;  // 2: rotary coeff
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV, IN_K_CACHE, IN_V_CACHE,
                                       IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = 1.0 / sqrt(param.dk);
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.isEncoder = true;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV,
                                     IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = 1.0 / sqrt(param.dk);
        paDeParam.kvHeadNum = param.headNum;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_K_CACHE, IN_V_CACHE, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMIDATE_ATTENTIONOUT};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    }

    atb_speed::common::ParallelParam selfOutLinearParam;
    selfOutLinearParam.rank = param.rank;
    selfOutLinearParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = false;
    selfOutLinearParam.transposeB = param.transposedWeight;
    selfOutLinearParam.backend = param.backend;
    atb_speed::common::RowParallelLinear(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_ATTENTIONOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    selfOutLinearNode.inTensorReshapeFuncs.resize(selfOutLinearNode.inTensorIds.size());
    selfOutLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;  // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    };

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    CreateOperation(rmsNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::common::MlpGateParam mlpParam;
    mlpParam.rank = param.rank;
    mlpParam.rankSize = param.rankSize;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.transposeB = param.transposedWeight;
    mlpParam.isBias = false;
    mlpParam.isPack = true;
    mlpParam.backend = param.backend;
    atb_speed::common::MlpGateLayer(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEUPWEIGHT, IN_MLPDOWNWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LAYEROUT};

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

FlashAttentionHostBinder::FlashAttentionHostBinder() {}

FlashAttentionHostBinder::~FlashAttentionHostBinder() {}

void FlashAttentionHostBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void FlashAttentionHostBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_INPUT_LENGTHS).hostData = seqLen_.data();
}
} // namespace llama_pa
} // namespace atb_speed