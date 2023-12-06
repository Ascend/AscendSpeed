#!/bin/bash

# The number of parameters is not aligned
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# modify config according to your own actual situation
CHECKPOINT="your model path"
TOKENIZER_PATH=./llama2-70b-hf/

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_llama.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 48  \
       --hidden-size 8192  \
       --ffn-hidden-size 22016 \
       --mlp-layer-fusion \
       --load ${CHECKPOINT}  \
       --num-attention-heads 64 \
       --position-embedding-type rope \
       --group-query-attention \
       --num-query-groups 8 \
       --max-position-embeddings 4096 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --use-flash-attn \
       --use-fused-rmsnorm \
       --seed 42 \
       --normalization RMSNorm \
