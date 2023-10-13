# This is an example: training llama2 using zero/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=./dataset/llama_text_document
CHECKPOINT_PATH=./ckpt

DS_CONFIG=deepspeed_config_7B.json
ZERO_STAGE=2
GLOBAL_BATCH=32
MICRO_BATCH=4

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

    "gradient_accumulation_steps": 1,
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

# Main script
deepspeed pretrain_llama.py \
       --checkpoint-activations \
       --use-fused-rotary-pos-emb \
       --triangle-attn \
       --DDP-impl local \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers 32 \
       --hidden-size 4096 \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path ./dataset/llama/ \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0003 \
       --lr-decay-style cosine \
       --min-lr 3.0e-5 \
       --weight-decay 1.0e-1 \
       --clip-grad 1.0 \
       --lr-warmup-iters 5000 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1.0e-5 \
       --initial-loss-scale 4096.0 \
       --use-cpu-initialization \
       $ds_args \
       --fp16 | tee logs/NPU_llama2_7b_shape_fp16_layer32_8p_pretrain.out

