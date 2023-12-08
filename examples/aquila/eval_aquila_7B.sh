# This is an example: training llama using PTD/
source /path/to/cann/ascend-toolkit/set_env.sh

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="./model_weights/aquila/"
VOCAB_FILE="./HF_Aquila7B_downloaded/"
DATA_PATH="./eval_datasets/boolq/test"
TASK="boolq"
# Different task needs different max_new_tokens value, please follow the instruction in readme.
python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation_llama.py \
       --task-data-path $DATA_PATH \
       --task $TASK \
       --seq-length 2048 \
       --max-new-tokens 1 \
       --max-position-embeddings 2048 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --load ${CHECKPOINT}  \
       --num-attention-heads 32  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $VOCAB_FILE \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --layernorm-epsilon 1e-6 \
       --make-vocab-size-divisible-by 8 \
       --use-flash-attn \
       --pad-vocab-size-to 100008 \
       --seed 42 | tee logs/train.log
