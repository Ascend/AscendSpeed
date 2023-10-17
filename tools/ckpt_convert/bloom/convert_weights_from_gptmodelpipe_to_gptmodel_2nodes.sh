#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
source /usr/local/Ascend/ascend-toolkit/set_env.sh

INPUT_PATH="your input checkpoint path"
OUTPUT_PATH="your output checkpoint path"

SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_gptmodelpipe_to_gptmodel_v2.py
python $SCRIPT_PATH \
    --input-model-dir ${INPUT_PATH} \
    --output-model-dir ${OUTPUT_PATH} \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --type 176B
