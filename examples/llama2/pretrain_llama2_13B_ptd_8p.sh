export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1
export INF_NAN_MODE_ENABLE=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH=./dataset/llama_text_document
LOAD_CHECKPOINT=./model/LLAMA-2-13B-hf_tp8_pp1
SAVE_CHECKPOINT=./model/LLAMA-2-13B-hf_tp8_pp1_save/
TOKENIZER_PATH=./model/LLAMA-2-13B-hf

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Main script
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
       --DDP-impl local \
       --is-instruction-dataset \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --sequence-parallel \
       --num-layers 40 \
       --hidden-size 5120 \
       --position-embedding-type rope \
       --normalization RMSNorm \
       --ffn-hidden-size 13824 \
       --num-attention-heads 40 \
       --micro-batch-size 2 \
       --global-batch-size 256 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --train-iters 5000 \
       --lr-warmup-iters 1000 \
       --save $SAVE_CHECKPOINT \
       --load $LOAD_CHECKPOINT \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 1e-6 \
       --lr-decay-style cosine \
       --min-lr 1e-8 \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --initial-loss-scale 4096.0 \
       --seed 1234 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1.0e-5 \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 100 \
       --use-fused-rotary-pos-emb \
       --use-flash-attn \
       --use-fused-rmsnorm \
       --bf16 | tee logs/train_13B_llama2_npu.log
