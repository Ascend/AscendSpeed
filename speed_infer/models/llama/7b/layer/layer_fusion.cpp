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
#include "layer_fusion.h"
#include "models/llama/7b/operation/position_embedding_fusion.h"
#include "models/llama/7b/operation/fusion_mlp.h"

namespace atb_speed {
namespace llama_7b {
enum LLaMA7BLayerTensorId {
    // hidden_states 0
    IN_HIDDENSTATES = 0,
    // norm_weight1
    IN_NORMWEIGHT,
    // linear_weight 2
    IN_MIXEDQKVLINEARWEIGHT,
    // out_weight 3
    IN_SELFOUTLINEARWEIGHT,
    // norm_weight 4
    IN_SELFOUTNORMWEIGHT,
    // mlp_gate_weight 5
    IN_MLPGATEUPWEIGHT,
    // mlp_down_weight 6
    IN_MLPDOWNWEIGHT,
    // cos 7
    IN_COSTABLE,
    // sin 8
    IN_SINTABLE,
    // attenion_mask 9
    IN_ATTENTIONMASK,
    // cache_k 10
    IN_CACHEK,
    // cache_v 11
    IN_CACHEV,
    // token_offset 12
    IN_TOKENOFFSET,
    // seq_len 13
    IN_SEQLEN,
    // layer_id 14
    IN_LAYERID,
    // decoder_output 15
    OUT_LLAMA7BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQKVLINEAROUT,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
};
static const uint64_t IN_TENSOR_COUNT = 15;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 10;
static const uint64_t NODE_COUNT = 9;

atb::Status FusionLayerOperation(const LayerFusionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "FusionLayer";
    size_t nodeId = 0;
    // rmsnorm 0
    atb::Node &inputNormNode = opGraph.nodes.at(nodeId++);
    // query_linear 1
    atb::Node &mixdQKVLinearNode = opGraph.nodes.at(nodeId++);
    // position_enbedding 2
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);
    // atb::Node &mulsQNode = opGraph.nodes.at(nodeId++);
    // kvcache_attention 3
    atb::Node &selfAttentionKvCacheNode = opGraph.nodes.at(nodeId++);
    // out_linear 4
    atb::Node &selfOutLinearNode = opGraph.nodes.at(nodeId++);
    // residual 5
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    // norm 6
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    // mlp 7
    atb::Node &mlpNode = opGraph.nodes.at(nodeId++);
    // mlpresidual_add 8
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(inputNormParam, &inputNormNode.operation);
    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    // (bsz,seq_len,hidden_size) - > (bsz,seq_len,hidden_size)
    atb::infer::LinearParam mixdQLinearParam = {false, false, false};
    mixdQLinearParam.hasBias = false;
    CreateOperation(mixdQLinearParam, &mixdQKVLinearNode.operation);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_MIXEDQKVLINEARWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQKVLINEAROUT};

    atb_speed::llama_7b::PositionEmbedding1dFusionParam positionEmbedding1dFusionParam;
    positionEmbedding1dFusionParam.rotaryCoeff = param.rotaryCoeff;
    positionEmbedding1dFusionParam.headNum = param.headNum;
    atb_speed::llama_7b::PositionEmbeddingFusionOperation(positionEmbedding1dFusionParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQKVLINEAROUT, IN_COSTABLE, IN_SINTABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV};
    
    atb::infer::SelfAttentionParam selfattentionparam;
    selfattentionparam.headDim = param.dk;
    selfattentionparam.headNum = param.headNum;
    selfattentionparam.qkScale = 1.0 / sqrt(param.dk);
    CreateOperation(selfattentionparam, &selfAttentionKvCacheNode.operation);

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

    atb::infer::LinearParam selfOutLinearParam;
    selfOutLinearParam.hasBias = false;
    CreateOperation(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer::ElewiseParam selfResidualAddParam;
    selfResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(selfResidualAddParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    atb_speed::llama_7b::FusionMlpParam mlpParam;
    atb_speed::llama_7b::FusionMlp(mlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEUPWEIGHT, IN_MLPDOWNWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA7BLAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

LayerFusionBinder::LayerFusionBinder() {}

LayerFusionBinder::~LayerFusionBinder() {}

void LayerFusionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset_.push_back(item.get<int>());
    }

    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    layerId_ = paramJson["layerId"].get<int>();

    ATB_LOG(INFO) << "FusionLayerOperation ParseParam tokenOffset:" <<
                      tokenOffset_ << ", seqLen:" << seqLen_ << ",layerId_";
}

void LayerFusionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = 16;
    const uint32_t seqLenTensorId = 17;
    const uint32_t layerIdTensorId = 18;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(layerIdTensorId).hostData = &layerId_;
}
} // namespace llama_7b
} // namespace atb_speed