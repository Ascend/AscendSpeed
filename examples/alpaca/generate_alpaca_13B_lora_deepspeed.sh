#!/bin/bash

export TOKENIZERS_PARALLELISM=false

NNODES=1
NPUS_PER_NODE=8

CHECKPOINT="your origin deepspeed checkpoint path (TP=1, PP=1)"
LORA_CHECKPOINT="your lora checkpoint path"
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
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

deepspeed --num_nodes $NNODES --num_gpus $NPUS_PER_NODE \
       ./tasks/inference/inference_alpaca.py \
       --no-contiguous-buffers-in-local-ddp \
       --num-layers 40  \
       --hidden-size 5120  \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40  \
       --max-position-embeddings 2048 \
       --tokenizer-type PretrainedFromHF  \
       --load "${CHECKPOINT}"  \
       --lora-load "${LORA_CHECKPOINT}" \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --max-new-tokens 256 \
       --seed 42 \
       --lora-r 16 \
       --lora-alpha 32 \
       --lora-target-modules query_key_value dense gate_proj up_proj down_proj \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --no-pipeline-parallel \
