# This is an example: train llama using TD,
# the number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /path/to/cann/ascend-toolkit/set_env.sh

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA=./dataset/llama_text_document
CHECKPOINT=./ckpt

DS_CONFIG=deepspeed_config_13B.json
ZERO_STAGE=3
GLOBAL_BATCH=64
MICRO_BATCH=2

export INF_NAN_MODE_ENABLE=1

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
        "allgather_bucket_size": 1e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 1e8,
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
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT \
       --data-path $DATA \
       --tokenizer-name-or-path ./dataset/llama/ \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       $ds_args \
       --fp16 | tee logs/train_13B.log
