# This is an example: train llama using PTD.
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=./dataset/llama_text_document
CHECKPOINT=./model_weights/llama-13b
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 8 \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size 1 \
       --global-batch-size 128 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 1000 \
       --lr-decay-iters 640 \
       --load $CHECKPOINT \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path ./dataset/llama/ \
       --tokenizer-not-use-fast \
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
       --checkpoint-policy custom \
       --recomputation-layer-num 3 2 1 0 0 0 0 0 \
       --triangle-attn \
       --use-fused-rotary-pos-emb \
       --fp16 | tee logs/train_13B.log
