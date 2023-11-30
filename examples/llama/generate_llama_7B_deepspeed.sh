#!/bin/bash

export TOKENIZERS_PARALLELISM=false

NNODES=1
NPUS_PER_NODE=8

CHECKPOINT="your megatron checkpoint path"
VOCAB_FILE="your vocab path"

ZERO_STAGE=0
MICRO_BATCH_SIZE=1
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 8
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

deepspeed --num_nodes $NNODES --num_gpus $NPUS_PER_NODE \
       ./tasks/inference/inference_llama.py \
       --no-contiguous-buffers-in-local-ddp \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --load "${CHECKPOINT}"  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --seq-length 2048 \
       --max-new-tokens 64 \
       --seed 42 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --no-pipeline-parallel \
       --position-embedding-type rope \
       --normalization RMSNorm \
