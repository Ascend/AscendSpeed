# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="Your ckpt file path"
VOCAB_FILE="Your vocab file path"
DATA_PATH="../dataset/boolq/test"
TASK="boolq"
# Different task needs different max_new_tokens value, please follow the instruction in readme.
python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation_llama.py   \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 1024 \
       --max-new-tokens 2 \
       --max-position-embeddings 1024 \
       --tensor-model-parallel-size 4  \
       --pipeline-model-parallel-size 2  \
       --num-layers 60 \
       --hidden-size 6656  \
       --ffn-hidden-size 17920 \
       --load ${CHECKPOINT}  \
       --num-attention-heads 52  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --mlp-layer-fusion \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --seed 42 | tee logs/evaluation.log
