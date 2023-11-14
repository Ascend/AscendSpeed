# This is an example: train llama using PTD,
# the number of parameters is not aligned

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6013
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/dataset/enwiki-gpt/gpt_text_sentence
CHECKPOINT_PATH=./ckpt_llama

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
basepath=$(cd `dirname $0`; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/run_llama_ptd.py \
       --DDP-impl local \
       --foldx-mode "fifo" \
       --use-distributed-optimizer \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 8 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 5 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /home/dataset/gpt2-vocab.json \
       --merge-file /home/dataset/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --sequence-parallel \
       --fp16
