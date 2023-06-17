# This is an example: train gpt using PTD,
# the number of parameters is not aligned

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

CHECKPOINT_PATH='./ckpt'
TENSORBOARD_PATH='./tensorboard/'
LOGS_PATH='./logs/'
mkdir -p $LOGS_PATH

# train parameter 
MASTER_ADDR=localhost
MASTER_PORT=5998
GPUS_PER_NODE=8
NNODES=1

# dataset path
TOKENIZER_NAME_OR_PATH=/home/dataset/bloom_vocab/vocab_file
DATA_PATH=/home/dataset/enwiki-gpt/gpt_text_sentence

config_json="./ds_config.json"

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": 4,
  "train_batch_size": 16,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 0
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

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
    /usr1/workspace/PyTorch_PR_AscendSpeed_master/CODE/tests/st/test_bloom/run_bloom_ptd.py \
    --tokenizer-type PretrainedFromHF \
    --embed-layernorm \
    --tokenizer-name-or-path $TOKENIZER_NAME_OR_PATH \
    --data-path $DATA_PATH \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --pad-vocab-size-to 250880 \
    --train-iters 5 \
    --lr-decay-iters 320000 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 8 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
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
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --log-interval 1 \
    --save-interval 250 \
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
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage 0 \
    --deepspeed-activation-checkpointing  \
    --distributed-backend nccl
