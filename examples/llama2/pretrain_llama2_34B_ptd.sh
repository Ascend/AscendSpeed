# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export COMBINED_ENABLE=1
export MULTI_STREAM_MEMORY_REUSE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=2
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=./dataset/llama_text_document
TOKENIZER_PATH=./tokenizer/
CHECKPOINT_LOAD_PATH=./load_ckpt
CHECKPOINT_SAVE_PATH=./save_ckpt
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
nohup python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --sequence-parallel \
       --use-flash-attn \
       --mlp-layer-fusion \
       --use-fused-rmsnorm \
       --release-fp32-grad \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 2 \
       --num-layers 48 \
       --hidden-size 8192 \
       --ffn-hidden-size 22016 \
       --num-attention-heads 64 \
       --normalization RMSNorm \
       --position-embedding-type rope \
       --group-query-attention \
       --num-query-groups 8 \
       --micro-batch-size 2 \
       --global-batch-size 512 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 1000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_SAVE_PATH \
       --load $CHECKPOINT_LOAD_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --pad-vocab-size-to 32000 \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --bf16 | tee logs/train.log &
