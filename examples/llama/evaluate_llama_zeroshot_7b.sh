#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORLD_SIZE=8
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TASK="HellaSwag"
VALID_DATA=.dataset/llama_zero_shot/hellaswag/hellaswag_val.jsonl


CHECKPOINT=./ckpt

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type PretrainedFromHF \
               --tokenizer-name-or-path ./dataset/llama/  \
               --tokenizer-not-use-fast \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 8 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --log-interval 1 \
               --layernorm-epsilon 1e-6 \
               --fp16 \
               --no-load-optim \
               --no-load-rng \
               --position-embedding-type rope \
               --normalization RMSNorm \
