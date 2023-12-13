#! /bin/bash

export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="Your ckpt file path"
VOCAB_FILE="Your vocab file path"
DATA_PATH="../dataset/boolq/test"
TASK="boolq"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/evaluation/evaluation_llama.py   \
       --no-contiguous-buffers-in-local-ddp \
       --task-data-path $DATA_PATH \
       --task $TASK \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 80 \
       --hidden-size 8192  \
       --ffn-hidden-size 22016 \
       --load "${CHECKPOINT}"  \
       --num-attention-heads 64  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 2048 \
       --max-new-tokens 2 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --seed 42 | tee logs/evaluation.log
