#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# output data path
CHECKPOINT_PATH='./ckpt'
TENSORBOARD_PATH='./tensorboard/'
LOGS_PATH='./logs/'
mkdir -p $LOGS_PATH

# train parameter 
MASTER_ADDR=localhost
MASTER_PORT=5999
GPUS_PER_NODE=8
NNODES=1

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=512

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048

SAVE_INTERVAL=250

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens

# dataset path
TOKENIZER_NAME_OR_PATH=/home/bloom_data/vocab_file/
DATA_PATH=/home/bloom_data/oscar_data_1g/my-gpt2_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --max_restarts 0 --tee 3"

TRANSFORMERS_OFFLINE=1  \
    python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_bloom.py \
    --tokenizer-type PretrainedFromHF \
    --embed-layernorm \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --data-path $DATA_PATH \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --pad-vocab-size-to 250880 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 192 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --init-method-std 0.0048 \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1.2e-4 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 1 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-impl mmap \
    --distributed-backend nccl \
    --sequence-parallel
    