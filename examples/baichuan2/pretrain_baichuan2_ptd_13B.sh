# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200

# Change for multinode config
MASTER_ADDR=xx.xx.x.xx
MASTER_PORT=12892
NNODES=2
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
GLOBAL_BATCH=256
MICRO_BATCH=2

DATA_PATH=./data/baichuan2_txt
TOKENIZER_PATH=./tokenizer

CHECKPOINT_PATH=./ckpt
LOAD_PATH=./weight

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
rm -rf kernel_meta*

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_baichuan.py \
       --DDP-impl local \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --sequence-parallel \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13696 \
       --num-attention-heads 40 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length 4096 \
       --normalization RMSNorm \
       --use-fused-rmsnorm \
       --max-position-embeddings 4096 \
       --train-iters 1000 \
       --save $CHECKPOINT_PATH \
       --load $LOAD_PATH \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --make-vocab-size-divisible-by 8 \
       --distributed-backend nccl \
       --lr 2e-5 \
       --lr-decay-style constant \
       --min-lr 1e-8 \
       --weight-decay 1e-4 \
       --position-embedding-type alibi \
       --clip-grad 1.0 \
       --layernorm-epsilon 1e-6 \
       --initial-loss-scale 8188.0 \
       --z-loss-weight 0 \
       --lm-norm-weight \
       --keep-last-token \
       --square-alibi-mask \
       --release-fp32-grad \
       --mlp-layer-fusion \
       --use-distributed-optimizer \
       --use-flash-attn \
       --auto-recompute-device-size 57344 \
       --seed 1234 \
       --adam-beta1 0.9 \
       --adam-beta2 0.98 \
       --adam-eps 1e-8 \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --bf16 | tee logs/loss_${logfile}.log
       