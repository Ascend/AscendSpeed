#!/bin/bash

source env_npu.sh
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

six_ALL_CCFRWORK=$(pwd)
out_path=$six_ALL_CCFRWORK/data/
variant=main

# 保存相关配置
DATA_OUTPUT_PATH=$out_path/checkpoints/tr11d-7B1-ml
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/tr11d-7B1-ml-logs
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant
LOGS_PATH=$REPO_PATH/logs/$variant
mkdir -p $LOGS_PATH

# 训练数据文件配置
MEGATRON_DEEPSPEED_REPO=$six_ALL_CCFRWORK/Megatron-DeepSpeed
#cd $MEGATRON_DEEPSPEED_REPO

BIGSCIENCE_REPO=$six_ALL_CCFRWORK/bigscience
TRAIN_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/train_data.txt
VALID_DATA_PATH=$MEGATRON_DEEPSPEED_REPO/data/valid_data.txt
CATALOGUE_JSON_PATH=$BIGSCIENCE_REPO/data/catalogue/training_dataset_ratios_merged_nigercongo_v3.json
LOAD_RATIOS_SCRIPT=$BIGSCIENCE_REPO/data/catalogue/load_ratios_meg_ds_format.py
# python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split train --output-meg-ds-ratio-file $TRAIN_DATA_PATH
# python $LOAD_RATIOS_SCRIPT --dataset-ratios-path $CATALOGUE_JSON_PATH --split valid --output-meg-ds-ratio-file $VALID_DATA_PATH

# 训练参数配置
MASTER_ADDR=localhost
MASTER_PORT=8082
GPUS_PER_NODE=8
NNODES=1
PP_SIZE=1
TP_SIZE=8
DP_SIZE=$((NNODES*GPUS_PER_NODE/(PP_SIZE*TP_SIZE)))

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512

NLAYERS=30
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048

SAVE_INTERVAL=250

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens

TOKENIZER_NAME_OR_PATH=/home/wangyixian/data/vocab_file
DATA_PATH=/home/wangyixian/oscar_data_1g/my-gpt2_text_document

ZERO_STAGE=0 # important: bf16 must use z0! it implements its own zero stage 1 equivalent
config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
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

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --max_restarts 0 --tee 3"

    #--abort-on-unmet-fused-kernel-constraints \
    #--train-weighted-split-paths-path $TRAIN_DATA_PATH \
    #--valid-weighted-split-paths-path $VALID_DATA_PATH \
    # --embed-layernorm \
    #--deepspeed \
    #--deepspeed_config ${config_json} \
    #--zero-stage ${ZERO_STAGE} \
    #--deepspeed-activation-checkpointing 
TRANSFORMERS_OFFLINE=1  \
    python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --data-path $DATA_PATH \
    --pad-vocab-size-to 250880 \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 192 16 9_765_625 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --init-method-std 0.0048 \
    --fp16 \
    --seed 42 \
    --position-embedding-type alibi \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1.2e-4 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --exit-duration-in-mins 5990 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-impl mmap \
    --distributed-backend nccl