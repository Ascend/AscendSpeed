# This is an example: training llama using PTD/

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=12892
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DATA_PATH=./dataset_baichuan2-7B/alpaca_text_document
TOKENIZER_PATH=./baichuan2-7B-hf/
CHECKPOINT_PATH=./ckpt
LOAD_PATH=./weight

MICRO_BATCH=4
GLOBAL_BATCH=128

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
rm -rf kernel_meta*

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_baichuan.py \
  --DDP-impl local \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 1 \
  --sequence-parallel \
  --num-layers 32 \
  --hidden-size 4096 \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --ffn-hidden-size 11008 \
  --num-attention-heads 32 \
  --micro-batch-size $MICRO_BATCH \
  --global-batch-size $GLOBAL_BATCH \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --train-iters 100000 \
  --save $CHECKPOINT_PATH \
  --load $LOAD_PATH \
  --data-path $DATA_PATH \
  --tokenizer-name-or-path $TOKENIZER_PATH \
  --tokenizer-not-use-fast \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 1e-6 \
  --lr-decay-style cosine \
  --min-lr 1e-8 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --seed 1234 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --mlp-layer-fusion \
  --use-flash-attn \
  --use-fused-rotary-pos-emb \
  --use-fused-rmsnorm \
  --lm-norm-weight \
  --bf16 | tee logs/loss_${logfile}.log
