#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
source /usr/local/Ascend/ascend-toolkit/set_env.sh

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


VOCAB_FILE="your VOCAB FILE path"
CHECKPOINT="your checkpoint path"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_gpt.py \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 30  \
       --hidden-size 4096  \
       --num-attention-heads 32 \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --max-new-tokens 256 \
       --seed 42 \
       --load "${CHECKPOINT}"  \
       --embed-layernorm \
       --position-embedding-type alibi \
       --no-add-gate \
       --add-bias-linear \
       --query-key-layer-scaling \
       --no-attention-softmax-in-fp32 \
       --no-untie-embeddings-and-output-weights
