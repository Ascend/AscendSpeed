#!/bin/bash

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./chinese_alpaca_plus_lora_13b \
    --output-model-dir ./chinese_alpaca_plus_lora_13b_deepspeed \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --type 13B \
    --deepspeed \

