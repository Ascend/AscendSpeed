#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:/home/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1600
# Enable memory reuse in INF_NAN mode can reduce memory usage and achieve lossless performance
export MULTI_STREAM_MEMORY_REUSE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# If this environment variable is set, all nodes will compile the dataset,
# which is suitable for cluster training.
export AZUREML_EXPERIMENT_ID=0

# output data path
CHECKPOINT_PATH='./ckpt'
TENSORBOARD_PATH='./tensorboard/'
LOGS_PATH='./logs/'
mkdir -p $LOGS_PATH

# train parameter 
MASTER_ADDR=localhost
MASTER_PORT=12890
GPUS_PER_NODE=8
NNODES=12
NODE_RANK=0
PP_SIZE=12
TP_SIZE=8

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2048

NLAYERS=70
NHIDDEN=14336
NHEADS=112
SEQ_LEN=2048

SAVE_INTERVAL=5000

TRAIN_SAMPLES=220_000_000  # 450B tokens
LR_DECAY_SAMPLES=200_000_000  # Decay for the first 410B tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=183_105  # 375M tokens

# dataset path
TOKENIZER_NAME_OR_PATH=/home/bloom_data/vocab_file/
DATA_PATH=/home/bloom_data/oscar_data_1g/my-gpt2_text_document

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
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TRANSFORMERS_OFFLINE=1  \
    python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_bloom.py \
    --tokenizer-type PretrainedFromHF \
    --embed-layernorm \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --data-path $DATA_PATH \
    --attention-dropout 0 \
    --hidden-dropout 0 \
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
    --bf16 \
    --seed 42 \
    --position-embedding-type alibi \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --log-interval 1 \
    --save $CHECKPOINT_PATH \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 1 \
    --tensorboard-queue-size 5 \
    --load $CHECKPOINT_PATH \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --data-impl mmap \
    --distributed-backend nccl \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing  \
    --sequence-parallel \
    --checkpoint-activations \
    --use-manual-layer-allocation \
    --manual-layers 5,6,6,6,6,6,6,6,6,6,6,5

