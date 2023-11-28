#!/bin/bash
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

CHECKPOINT="your megatron checkpoint path"
LORA_CHECKPOINT="your lora checkpoint path"
VOCAB_FILE="your vocab path"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ./tasks/inference/inference_llama.py \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load "${CHECKPOINT}"  \
       --lora-load "${LORA_CHECKPOINT}" \
       --num-attention-heads 32  \
       --seq-length 1024 \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --max-new-tokens 256 \
       --seed 42 \
       --lora-r 16 \
       --lora-alpha 32 \
       --lora-target-modules query_key_value dense gate_proj up_proj down_proj \
       --position-embedding-type rope \
       --normalization RMSNorm \
