#!/bin/bash

# Environment
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Distributed setting
DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr localhost --master_port 6001"

CHECKPOINTS_PATH=<checkpoints-path>
TOKENIZER_PATH=<tokenizer-vocab-file-path>

# Real script
python -m torch.distributed.run $DISTRIBUTED_ARGS ./examples/bloom_task/generate_bloom.py \
               --load CHECKPOINTS_PATH \
               --tokenizer-type PretrainedFromHF \
               --tokenizer-name-or-path TOKENIZER_PATH  \
               --tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --embed-layernorm \
               --position-embedding-type alibi \
               --num-layers 30 \
               --hidden-size 4096 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --init-method-std 0.0048 \
               --log-interval 1 \
               --layernorm-epsilon 1e-6 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --out-seq-length 1024 \
               --temperature 1.0 \
               --top_p 0.9 \
               --recompute
