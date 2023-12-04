# This is an example: train llama using PTD,
# the number of parameters is not aligned

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/dataset/alpaca_cn/alpaca
TOKENIZER_PATH=/home/dataset/llama_tokenizer
CHECKPOINT_PATH=./ckpt_llama
LORA_CHECKPOINT_PATH=./ckpt_llama_lora

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
basepath=$(cd `dirname $0`; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH
# save base model
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/run_llama_ptd.py \
       --DDP-impl local \
       --is-instruction-dataset \
       --variable-seq-lengths \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 2 \
       --global-batch-size 8 \
       --seq-length 96 \
       --max-position-embeddings 2048 \
       --train-iters 2 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-not-use-fast \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
       --recompute-method block \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --fp16 \
       --position-embedding-type rope \
       --normalization RMSNorm \

# save lora adapter
MASTER_PORT=6016
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/run_llama_ptd.py \
       --DDP-impl local \
       --is-instruction-dataset \
       --variable-seq-lengths \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 2 \
       --global-batch-size 8 \
       --seq-length 96 \
       --max-position-embeddings 2048 \
       --train-iters 2 \
       --lr-decay-iters 320000 \
       --save $LORA_CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-not-use-fast \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
       --recompute-method block \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
       --lora-modules-to-save word_embeddings output_layer \
       --fp16 \
       --position-embedding-type rope \
       --normalization RMSNorm \

# load lora adapter
MASTER_PORT=6017
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/run_llama_ptd.py \
       --DDP-impl local \
       --is-instruction-dataset \
       --variable-seq-lengths \
       --no-contiguous-buffers-in-local-ddp \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 2 \
       --global-batch-size 8 \
       --seq-length 96 \
       --max-position-embeddings 2048 \
       --train-iters 2 \
       --lr-decay-iters 320000 \
       --save $LORA_CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --lora-load $LORA_CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tokenizer-not-use-fast \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
       --recompute-method block \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 1 \
       --lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
       --lora-modules-to-save word_embeddings output_layer \
       --fp16 \
       --position-embedding-type rope \
       --normalization RMSNorm \
