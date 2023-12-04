# This is an example: training llama using PTD/

# The number of parameters is not aligned
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export TASK_QUEUE_ENABLE=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NPUS_PER_NODE=8

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

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

# Real script
python -m torch.distributed.run $DISTRIBUTED_ARGS \
       ./tasks/evaluation/evaluation_bloom.py \
       --no-contiguous-buffers-in-local-ddp \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --embed-layernorm \
       --seq-length 2048 \
       --max-new-tokens 2 \
       --position-embedding-type alibi \
       --load ${CHECKPOINT}  \
       --max-position-embeddings 2048 \
       --pad-vocab-size-to 250880 \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 70  \
       --hidden-size 14336  \
       --num-attention-heads 112 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --fp16 \
       --micro-batch-size $MICRO_BATCH_SIZE  \
       --init-method-std 0.0048 \
       --layernorm-epsilon 1e-6 \
       --fp16 \
       --no-load-optim \
       --no-load-rng \
       --no-add-gate \
       --add-bias-linear \
       --query-key-layer-scaling \
       --no-attention-softmax-in-fp32 \
       --no-untie-embeddings-and-output-weights \
       --seed 42 | tee logs/train.log

