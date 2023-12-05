# This is an example: train llama using PTD.
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=16
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=<data-path>
LOAD_CHECKPOINT_PATH=<origin-ckpt-path>
SAVE_CHECKPOINT_PATH=<ckpt-path>
TOKENIZER_PATH=<tokenizer-path>

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
      pretrain_llama.py \
      --DDP-impl local \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 2 \
      --num-layers 40 \
      --hidden-size 5120 \
      --position-embedding-type rope \
      --normalization RMSNorm \
      --ffn-hidden-size 13824 \
      --num-attention-heads 40 \
      --micro-batch-size 1 \
      --global-batch-size 512 \
      --seq-length 2048 \
      --max-position-embeddings 2048 \
      --train-iters 5000 \
      --lr-decay-iters 5000 \
      --load $LOAD_CHECKPOINT_PATH \
      --data-path $DATA_PATH \
      --tokenizer-name-or-path $TOKENIZER_PATH \
      --tokenizer-not-use-fast \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --data-impl mmap \
      --split 949,50,1 \
      --distributed-backend nccl \
      --lr 1.0e-6 \
      --lr-decay-style cosine \
      --min-lr 1.0e-7 \
      --weight-decay 1e-2 \
      --clip-grad 1.0 \
      --lr-warmup-fraction .01 \
      --log-interval 1 \
      --save-interval 10000 \
      --eval-interval 1000 \
      --eval-iters 10 \
      --initial-loss-scale 4096.0 \
      --checkpoint-activations \
      --use-fused-rotary-pos-emb \
      --use-flash-attn \
      --use-distributed-optimizer \
      --use-fused-rmsnorm \
      --fp16 | tee logs/train_llama_13B.log