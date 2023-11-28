#!/bin/bash
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

CHECKPOINT="your megatron checkpoint path"
VOCAB_FILE="your vocab path"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/inference/inference_llama.py \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 2  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load "${CHECKPOINT}"  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --max-new-tokens 256 \
       --seed 42 \
       --position-embedding-type rope \
       --normalization RMSNorm \

