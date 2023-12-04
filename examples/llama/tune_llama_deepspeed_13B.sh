# This is an example: train llama using TD,

# the number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/home/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TP=1
PP=1

DATA_PATH=<data-path>
LOAD_CHECKPOINT_PATH=<origin-ckpt-path>
SAVE_CHECKPOINT_PATH=<ckpt-path>
TOKENIZER_PATH=<tokenizer-path>

DS_CONFIG=deepspeed_config_13B.json
ZERO_STAGE=2

MICRO_BATCH=4
GRADIENT_ACCUMULATION_STEP=4
GLOBAL_BATCH=$(($MICRO_BATCH * $GRADIENT_ACCUMULATION_STEP * $WORLD_SIZE))
EPOCH=2
TRAIN_ITERS=$((52000 / $GLOBAL_BATCH * $EPOCH))
echo $TRAIN_ITERS
SAVE_INTERVAL=$(($TRAIN_ITERS / 4))
echo $SAVE_INTERVAL

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

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

    "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEP},
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
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 256 \
       --max-position-embeddings 2048 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --train-iters ${TRAIN_ITERS} \
       --lr-decay-iters ${TRAIN_ITERS} \
       --save $SAVE_CHECKPOINT_PATH \
       --load $LOAD_CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-5 \
       --lr-decay-style cosine \
       --min-lr 0 \
       --weight-decay 0. \
       --clip-grad 1.0 \
       --lr-warmup-iters 200 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval ${SAVE_INTERVAL} \
       --eval-interval 1000 \
       --eval-iters 10 \
       --use-cpu-initialization \
       --lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
       --lora-r 16 \
       --lora-alpha 32 \
       --is-instruction-dataset \
       $ds_args \
       --fp16 | tee logs/train_13B_deepspeed.log