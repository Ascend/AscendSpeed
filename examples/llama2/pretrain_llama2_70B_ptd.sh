# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=8
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=./dataset/llama_text_document
CHECKPOINT_LOAD_PATH=./load_ckpt
CHECKPOINT_SAVE_PATH=./save_ckpt
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --use-flash-attn \
       --sequence-parallel \
       --mlp-layer-fusion \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 8 \
       --num-layers 80 \
       --hidden-size 8192 \
       --ffn-hidden-size 28672 \
       --num-attention-heads 64 \
       --position-embedding-type rope \
       --group-query-attention \
       --num-query-groups 8 \
       --micro-batch-size 2 \
       --global-batch-size 1024 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 5000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_SAVE_PATH \
       --load $CHECKPOINT_LOAD_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path ./dataset/llama/ \
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
       --eval-interval 1000 \
       --eval-iters 10 \
       --release-fp32-grad \
       --bf16 | tee logs/train.log
