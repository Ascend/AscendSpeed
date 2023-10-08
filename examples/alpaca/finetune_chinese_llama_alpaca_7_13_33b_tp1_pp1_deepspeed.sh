# This script is used to run Chinese LLaMA Alpaca with 7B/13B/33B weights based on deepspeed launcher, configured with tensor model parallel size of 1, pipeline model parallel size of 1.
# add HCCL_OP_BASE_FFTS_MODE_ENABLE
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

# modify the script according to your own conda and ascend-toolkit path
export LD_LIBRARY_PATH=/usr/local/lib:/root/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# modify script orign dataset path and tokenizer path according to your own dataset path and tokenizer path
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH=<data-path>
# your own merged model path
MODEL_PATH=<model-path>

DS_CONFIG=deepspeed_config_13B.json
ZERO_STAGE=2
GLOBAL_BATCH=16
MICRO_BATCH=2

# 7b/13b/33b models use the following parameters respectively 
if [[ "$MODEL_PATH" == *13b* ]]; then
  num_layers=40
  hidden_size=5120
  ffn_hidden_size=13824
  num_heads=40
elif [[ "$MODEL_PATH" == *33b* ]]; then
  num_layers=60
  hidden_size=6656
  ffn_hidden_size=17920
  num_heads=52
else
  num_layers=32
  hidden_size=4096
  ffn_hidden_size=11008
  num_heads=32
fi

# This is the configuration for deepspeed
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

ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

deepspeed pretrain_llama.py \
         --DDP-impl local \
         --is-instruction-dataset \
         --tensor-model-parallel-size 1 \
         --pipeline-model-parallel-size 1 \
         --num-layers $num_layers \
         --hidden-size $hidden_size \
         --ffn-hidden-size $ffn_hidden_size \
         --num-attention-heads $num_heads \
         --micro-batch-size $MICRO_BATCH \
         --global-batch-size $GLOBAL_BATCH \
         --seq-length 1024 \
         --max-position-embeddings 2048 \
         --train-iters 1 \
         --lr-decay-iters 320000 \
         --load $MODEL_PATH \
         --data-path $DATA_PATH \
         --tokenizer-name-or-path $TOKENIZER_PATH \
         --tokenizer-not-use-fast \
         --data-impl mmap \
         --split 949,50,1 \
         --distributed-backend nccl \
         --lr 1e-4 \
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
         --lora-target-modules query_key_value dense gate_proj up_proj down_proj \
         --lora-r 64 \
         --lora-alpha 128 \
         --lora-modules-to-save word_embeddings lm_head.lm_head \
         $ds_args \
         --fp16 | tee logs/train.log
