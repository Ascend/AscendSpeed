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
#include "quant_decoder_parallel_layer_fa.h"
#include "models/llama/7b/operation/rope_fusion_operation.h"
#include "layers/mlp_gate_v2.h"
#include "layers/parallel_layer_v2.h"

namespace atb_speed {
namespace llama_13b {
const int ATTENTION_DIM_NUM = 4;
const int ATTENTION_DIM_2 = 2;
const int ATTENTION_DIM_3 = 3;

enum LayerQuantParallelFlashAttentionTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,

    IN_QMIXDWEIGHT,
    IN_QMIXD_DEQSCALE,
    IN_QMIXD_BIAS,

    IN_KMIXDWEIGHT,
    IN_KMIXD_DEQSCALE,
    IN_KMIXD_BIAS,

    IN_VMIXDWEIGHT,
    IN_VMIXD_DEQSCALE,
    IN_VMIXD_BIAS,

    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEAR_DEQSCALE,
    IN_SELFOUTLINEAR_BIAS,

    IN_SELFOUTNORMWEIGHT,

    IN_MLPGATEWEIGHT,
    IN_MLPGATE_DEQSCALE,
    IN_MLPGATE_BIAS,
    IN_MLPDOWNWEIGHT,
    IN_MLPDOWN_DEQSCALE,
    IN_MLPDOWN_BIAS,
    IN_MLPUPWEIGHT,
    IN_MLPUP_DEQSCALE,
    IN_MLPUP_BIAS,
    
    IN_POSITIONIDS,
    IN_COSTABLE,
    IN_SINTABLE,
    IN_ATTENTIONMASK,
    IN_CACHEK,
    IN_CACHEV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_BETA,
    IN_HOLDER,
    IN_LAYERID,

    OUT_LLAMA7BLAYEROUT,
    
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPLINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 35;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 11;
static const uint64_t NODE_COUNT = 11;

atb::Status QuantLayerParallelFlashAttentionOperation(const QuantLayerParallelFlashAttentionParam &param,
                                                      atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "LayerQuantParallelFlashAttention";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdKLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &mixdVLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    // RMSNORM量化
    atb::infer::RmsNormParam rmsNormParam;
    rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    rmsNormParam.normParam.quantInputScale = param.qkvInputScale;
    rmsNormParam.normParam.quantInputOffset = param.qkvInputOffset;
    rmsNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CreateOperation(rmsNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_BETA};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    // QKV LINEAR量化
    atb::infer::LinearQuantParam quantQkvLinearParam;
    quantQkvLinearParam.transposeB = true;
    CreateOperation(quantQkvLinearParam, &mixdQLinearNode.operation);
    mixdQLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QMIXDWEIGHT, IN_QMIXD_BIAS, IN_QMIXD_DEQSCALE};
    mixdQLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ};

    CreateOperation(quantQkvLinearParam, &mixdKLinearNode.operation);
    mixdKLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_KMIXDWEIGHT, IN_KMIXD_BIAS, IN_KMIXD_DEQSCALE};
    mixdKLinearNode.outTensorIds = {INTERMIDATE_MIXEDK};

    CreateOperation(quantQkvLinearParam, &mixdVLinearNode.operation);
    mixdVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_VMIXDWEIGHT, IN_VMIXD_BIAS, IN_VMIXD_DEQSCALE};
    mixdVLinearNode.outTensorIds = {INTERMIDATE_MIXEDV};

    atb_speed::llama_7b::RopeFusionParam ropeFusionParam;
    ropeFusionParam.headNum = param.headNum;
    atb_speed::llama_7b::RopeFusionOperation(ropeFusionParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_POSITIONIDS,
                            IN_COSTABLE,        IN_SINTABLE,        IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};

    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headDim = param.dk;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qScale = 1.0 / sqrt(param.dk);
    CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_CACHEK,
                                            IN_CACHEV,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = ATTENTION_DIM_NUM;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[ATTENTION_DIM_2] = param.headNum;
        newShape.dims[ATTENTION_DIM_3] = oldShape.dims[ATTENTION_DIM_2] / param.headNum;
    };

    // SelfAttention输出量化
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.isBias = true;
    selfOutLinearParam.isQuant = true;
    selfOutLinearParam.transposeB = true;
    selfOutLinearParam.quantParam.quantType = atb::infer::QUANT_INT8;
    selfOutLinearParam.quantParam.isQuantOp = true;
    selfOutLinearParam.quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutLinearParam.quantParam.inputScale = param.denseInputScale;
    selfOutLinearParam.quantParam.inputOffset = param.denseInputOffset;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT,
                                    IN_SELFOUTLINEAR_BIAS, IN_SELFOUTLINEAR_DEQSCALE, IN_HOLDER};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    // RMSNORM量化
    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.quantInputScale = param.selfLnInputScale;
    selfNormParam.normParam.quantInputOffset = param.selfLnInputOffset;
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_BETA};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    // MLP量化
    atb_speed::common::MlpGateParamV2 mlpParam;
    mlpParam.isBias=true;
    mlpParam.isPack=false;
    mlpParam.isQuant=true;
    mlpParam.transposeB=true;
    mlpParam.commDownParam.rank = param.rank;
    mlpParam.commDownParam.rankSize = param.rankSize;
    mlpParam.quantUpParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantUpParam.isQuantOp = false;
    mlpParam.quantGateParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.quantGateParam.isQuantOp = false;
    mlpParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    mlpParam.quantDownParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpParam.quantDownParam.inputScale = param.ffnOutInputScale;
    mlpParam.quantDownParam.inputOffset = param.ffnOutInputOffset;
    atb_speed::common::MlpGateLayerV2(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPUPWEIGHT, IN_MLPGATEWEIGHT, IN_MLPDOWNWEIGHT,
                           IN_MLPUP_DEQSCALE, IN_MLPGATE_DEQSCALE, IN_MLPDOWN_DEQSCALE,
                           IN_MLPUP_BIAS, IN_MLPGATE_BIAS, IN_MLPDOWN_BIAS,
                           IN_HOLDER, IN_HOLDER, IN_HOLDER};
    mlpNode.outTensorIds = {INTERMIDATE_MLPLINEAROUT};

    CreateOperation(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEAROUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA7BLAYEROUT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

QuantLayerPrallelFlashAttentionBinder::QuantLayerPrallelFlashAttentionBinder() {}

QuantLayerPrallelFlashAttentionBinder::~QuantLayerPrallelFlashAttentionBinder() {}

void QuantLayerPrallelFlashAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
}

void QuantLayerPrallelFlashAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = IN_TOKENOFFSET;
    const uint32_t seqLenTensorId = IN_SEQLEN;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
}

} // namespace llama_13b
} // namespace atb_speed