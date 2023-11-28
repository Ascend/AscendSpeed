#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
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
PP_SIZE=1
TP_SIZE=8

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16

NLAYERS=24
NHIDDEN=1024
NHEADS=16
SEQ_LEN=1024

SAVE_INTERVAL=250

# dataset path
TOKENIZER_NAME_OR_PATH=./dataset/bloom_vocab/vocab_file/
DATA_PATH=./dataset/oscar_data_1g/my-gpt2_text_document

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --max_restarts 0 --tee 3"

TRANSFORMERS_OFFLINE=1  \
    python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_bloom.py \
    --tokenizer-type PretrainedFromHF \
    --embed-layernorm \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --data-path $DATA_PATH \
    --pad-vocab-size-to 250880 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --init-method-std 0.0048 \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 3e-4 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --train-iters 5 \
    --lr-decay-iters 320000 \
    --eval-interval 1000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-impl mmap \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing  \
    --no-add-gate \
    --add-bias-linear \
    --query-key-layer-scaling \
    --no-attention-softmax-in-fp32 \
    --no-untie-embeddings-and-output-weights \
    --distributed-backend nccl | tee $LOGS_PATH/train.log
