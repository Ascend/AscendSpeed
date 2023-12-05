#!/bin/bash
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES_=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_}
ASCEND_RT_VISIBLE_DEVICES_ARRAY=(${CUDA_VISIBLE_DEVICES_//,/ })
echo "${ASCEND_RT_VISIBLE_DEVICES_ARRAY[@]}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

NPUS_PER_NODE=${#ASCEND_RT_VISIBLE_DEVICES_ARRAY[@]}    #获取数组的个数
echo NPUS_PER_NODE
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

TP=2
PP=2

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

ORIGIN_CHECKPOINT_PATH=<origin_checkpoint_path>
LORA_CHECKPOINT_PATH=<lora_checkpoint_path>
VOCAB_FILE=<vocab_file_path>
MERGED_MODEL_SAVE_PATH=<merged_model_save_path>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tools/lora/merge_lora.py \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size ${TP}  \
       --pipeline-model-parallel-size ${PP}  \
       --num-layers 40  \
       --hidden-size 5120  \
       --ffn-hidden-size 13696 \
       --num-attention-heads 40  \
       --save ${MERGED_MODEL_SAVE_PATH} \
       --load "${ORIGIN_CHECKPOINT_PATH}"  \
       --lora-load ${LORA_CHECKPOINT_PATH} \
       --lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
       --lora-r 16 \
       --lora-alpha 32 \
       --max-position-embeddings 4096 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --save-interval 1000 \
       --seed 42 \
       --position-embedding-type rope \
       --normalization RMSNorm
