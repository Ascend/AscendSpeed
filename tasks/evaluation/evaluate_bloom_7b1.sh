# This is an example: training llama using PTD/

# The number of parameters is not aligned
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
source /usr/local/Ascend/set_env.sh
export INF_NAN_MODE_ENABLE=0

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MICRO_BATCH_SIZE=1
CHECKPOINT="Your ckpt file path"
VOCAB_FILE="Your vocab file path"
DATA_PATH="../dataset/mmlu/test ../dataset/gsm8k/test ../dataset/boolq/test ../dataset/ceval/test"
TASK="boolq gsm8k mmlu ceval agieval bbh human_eval"
# Different task needs different max_new_tokens value, please follow the instruction in readme.

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

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

# Real script
deepspeed --num_nodes $NNODES --num_gpus $NPUS_PER_NODE \
       ./tasks/evaluation/evaluation_bloom.py \
       --no-contiguous-buffers-in-local-ddp \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 30  \
       --hidden-size 4096  \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --position-embedding-type alibi \
       --embed-layernorm \
       --tokenizer-type PretrainedFromHF  \
       --load "${CHECKPOINT}"  \
       --tokenizer-name-or-path "$VOCAB_FILE" \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --max-new-tokens 2 \
       --seed 42 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --no-add-gate \
       --add-bias-linear \
       --query-key-layer-scaling \
       --no-attention-softmax-in-fp32 \
       --no-untie-embeddings-and-output-weights \
       --seed 42 | tee logs/train.log

