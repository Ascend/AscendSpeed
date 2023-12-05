# This script is used to run Chinese LLaMA Alpaca with 7B/13B/33B weights, configured with tensor model parallel size of 4, pipeline model parallel size of 2.
# modify the script according to your own conda and ascend-toolkit path
export LD_LIBRARY_PATH=/usr/local/lib:/root/anaconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export INF_NAN_MODE_ENABLE=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

NPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# modify script orign dataset path and tokenizer path according to your own dataset path and tokenizer path
TOKENIZER_PATH=<tokenizer-path>
DATA_PATH=<data-path>
# your own merged model path
MODEL_PATH=<model-path>

GLOBAL_BATCH=250
MICRO_BATCH=2

# Distributed setting
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# 7b/13b/33b models use the following parameters respectively 
if [[ "$MODEL_PATH" == *13[Bb]* ]]; then
  num_layers=40
  hidden_size=5120
  ffn_hidden_size=13824
  num_heads=40
elif [[ "$MODEL_PATH" == *33[Bb]* ]]; then
  num_layers=60
  hidden_size=6656
  ffn_hidden_size=17920
  num_heads=52
else
  num_layers=32
  hidden_size=4096
  ffn_hidden_size=11008
  num_heads=32
fi

python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
         pretrain_llama.py \
         --no-contiguous-buffers-in-local-ddp \
         --DDP-impl local \
         --is-instruction-dataset \
         --tensor-model-parallel-size 4 \
         --pipeline-model-parallel-size 2 \
         --num-layers $num_layers \
         --hidden-size $hidden_size \
         --ffn-hidden-size $ffn_hidden_size \
         --num-attention-heads $num_heads \
         --micro-batch-size $MICRO_BATCH \
         --global-batch-size $GLOBAL_BATCH \
         --seq-length 2048 \
         --max-position-embeddings 2048 \
         --train-iters 1000 \
         --lr-decay-iters 320000 \
         --load $MODEL_PATH \
         --data-path $DATA_PATH \
         --tokenizer-name-or-path $TOKENIZER_PATH \
         --tokenizer-not-use-fast \
         --data-impl mmap \
         --initial-loss-scale 4096 \
         --split 949,50,1 \
         --distributed-backend nccl \
         --lr 1e-4 \
         --lr-decay-style cosine \
         --min-lr 1.0e-5 \
         --weight-decay 1e-2 \
         --clip-grad 1.0 \
         --lr-warmup-fraction .01 \
         --eval-iters 10 \
         --log-interval 1 \
         --save-interval 10000 \
         --position-embedding-type rope \
         --normalization RMSNorm \
         --use-flash-attn \
         --use-fused-rmsnorm \
         --lora-target-modules query_key_value dense gate_proj dense_h_to_4h dense_4h_to_h \
         --lora-r 32 \
         --lora-alpha 64 \
         --fp16 | tee logs/train.log
