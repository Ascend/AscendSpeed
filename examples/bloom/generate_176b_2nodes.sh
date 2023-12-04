#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
source /usr/local/Ascend/ascend-toolkit/set_env.sh

MASTER_ADDR=**.**.**.**
MASTER_PORT=12890
NNODES=2
NPUS_PER_NODE=8
NODE_RANK=1

VOCAB_FILE="your VOCAB FILE path"
CHECKPOINT="your checkpoint path"

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Real script
python -m torch.distributed.run $DISTRIBUTED_ARGS ./tasks/inference/inference_gpt.py \
               --no-contiguous-buffers-in-local-ddp \
               --load ${CHECKPOINT} \
               --tokenizer-type PretrainedFromHF \
               --tokenizer-name-or-path ${VOCAB_FILE}  \
               --tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 2 \
               --embed-layernorm \
               --position-embedding-type alibi \
               --num-layers 70  \
               --hidden-size 14336  \
               --num-attention-heads 112 \
               --max-position-embeddings 2048 \
               --seq-length 2048 \
               --micro-batch-size 1 \
               --init-method-std 0.0048 \
               --layernorm-epsilon 1e-6 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --no-add-gate \
               --add-bias-linear \
               --query-key-layer-scaling \
               --no-attention-softmax-in-fp32 \
               --no-untie-embeddings-and-output-weights
