export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DATA_PATH=dataset/llama_text_document
CHECKPOINT_PATH=ckpt
TOKENIZER_PATH=tokenizer

DS_CONFIG=ds_config.json
ZERO_STAGE=2
MICRO_BATCH=1
GLOBAL_BATCH=8

rm -rf kernel_meta*

cat <<EOT >$DS_CONFIG
{
    "gradient_accumulation_steps": 1,
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu":$MICRO_BATCH,
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 8,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "eps": 1.0e-8,
        "betas": [
          0.9,
          0.95
        ],
        "weight_decay": 0.0
      }
    },

    "steps_per_print": 1,

    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "contiguous_gradients": false,
        "allgather_bucket_size": 1e8,
        "reduce_bucket_size": 1e8,
        "overlap_comm": true,
        "reduce_scatter": true
    }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

SEQ_LEN=4096

deepspeed  pretrain_baichuan.py \
  --DDP-impl local \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 32 \
  --hidden-size 4096 \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --ffn-hidden-size 11008 \
  --num-attention-heads 32 \
  --micro-batch-size $MICRO_BATCH \
  --global-batch-size $GLOBAL_BATCH \
  --seq-length $SEQ_LEN \
  --max-position-embeddings $SEQ_LEN \
  --train-iters 1024 \
  --data-path $DATA_PATH \
  --tokenizer-name-or-path $TOKENIZER_PATH \
  --tokenizer-not-use-fast \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00001 \
  --lr-decay-style cosine \
  --min-lr 1.0e-6 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .1 \
  --log-interval 1 \
  --save-interval 10000 \
  --eval-interval 1000 \
  --eval-iters 1 \
  --checkpoint-activations \
  --recompute-method uniform \
  --recompute-num-layers 30 \
  --triangle-attn \
  $ds_args \
  --fp16 | tee logs/train.log
