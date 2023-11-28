# This is an example: train llama using PTD.

# The number of parameters is not aligned
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0    #1
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MICRO_BATCH=4
GRADIENT_ACCUMULATION_STEP=4
GLOBAL_BATCH=$(($MICRO_BATCH * $GRADIENT_ACCUMULATION_STEP * $WORLD_SIZE))
EPOCH=5
TRAIN_ITERS=$((52000 / $GLOBAL_BATCH * $EPOCH))
echo $TRAIN_ITERS
SAVE_INTERVAL=$(($TRAIN_ITERS / 4))
echo $SAVE_INTERVAL

TP=4
PP=2

DATA_PATH=<data-path>
LOAD_CHECKPOINT_PATH=<origin-ckpt-path>
SAVE_CHECKPOINT_PATH=<ckpt-path>
TOKENIZER_PATH=<tokenizer-path>

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
       pretrain_llama.py \
       --DDP-impl local \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 40 \
       --hidden-size 5120 \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size ${MICRO_BATCH} \
       --global-batch-size ${GLOBAL_BATCH} \
       --seq-length 256 \
       --max-position-embeddings 2048 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --train-iters ${TRAIN_ITERS} \
       --save ${SAVE_CHECKPOINT_PATH} \
       --load ${LOAD_CHECKPOINT_PATH} \
       --checkpoint-activations \
       --data-path ${DATA_PATH} \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1e-6 \
       --lr-decay-style cosine \
       --lr-decay-iters ${TRAIN_ITERS} \
       --min-lr 0 \
       --weight-decay 0. \
       --clip-grad 1.0 \
       --lr-warmup-iters 200 \
       --log-interval 1 \
       --save-interval ${SAVE_INTERVAL} \
       --eval-interval 1000 \
       --eval-iters 10 \
       --initial-loss-scale 4096.0 \
       --seed 1234 \
       --zero-stage 2 \
       --is-instruction-dataset \
       --lora-target-modules query_key_value dense gate_proj up_proj down_proj \
       --lora-r 64 \
       --lora-alpha 128 \
       --lora-modules-to-save word_embeddings lm_head.lm_head \
       --fp16 | tee logs/train_13B_megatron.log

