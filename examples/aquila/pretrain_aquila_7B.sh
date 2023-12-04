# This is an example: train llama using TD,
# the number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export INF_NAN_MODE_ENABLE=1
source /path/to/cann/ascend-toolkit/set_env.sh

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA=./dataset/aquila_text_document
CHECKPOINT=./ckpt

DS_CONFIG=deepspeed_config_7B.json
ZERO_STAGE=2
GLOBAL_BATCH=64
MICRO_BATCH=2

cat <<EOT > $DS_CONFIG
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 8,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "Adam"
    },

    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": 4,
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu":$MICRO_BATCH,
    "zero_allow_untested_optimizer": true
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

deepspeed pretrain_llama.py \
       --DDP-impl local \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 32 \
       --hidden-size 4096 \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --train-iters 1024 \
       --save $CHECKPOINT \
       --data-path $DATA \
       --tokenizer-name-or-path ./dataset/aquila/ \
       --tokenizer-not-use-fast \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --init-method-std 0.01 \
       --split 900,50,50 \
       --distributed-backend nccl \
       --lr 1.0e-6 \
       --lr-decay-style cosine \
       --lr-warmup-fraction .01 \
       --min-lr 1.0e-6 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 10000 \
       --no-load-optim \
       --no-load-rng \
       --no-bias-gelu-fusion \
       --use-flash-attn \
       --layernorm-epsilon 1e-6 \
       --make-vocab-size-divisible-by 8 \
       --pad-vocab-size-to 100008 \
       --use-fused-rmsnorm \
       $ds_args \
       --fp16 | tee logs/train_7B.log
