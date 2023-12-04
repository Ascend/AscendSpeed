# This is an example: training llama using PTD/

# The number of parameters is not aligned
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# modify config according to your own actual situation
LOAD_CHECKPOINT_PATH="your init model load path"
SAVE_CHECKPOINT_PATH="your model ckpt save path"
TOKENIZER_PATH=./llama-2-7b-hf/  #tokenizer path
DATA_PATH=./finetune_dataset/alpaca  #processed dataset

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
logfile=$(date +%Y%m%d)_$(date +%H%M%S)

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_llama.py \
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
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --init-method-std 0.01 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --data-path $DATA_PATH \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --tokenizer-not-use-fast \
       --split 100,0,0 \
       --distributed-backend nccl \
       --lr 1.25e-5 \
       --min-lr 1.25e-6 \
       --lr-decay-style cosine \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --initial-loss-scale 65536.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --load ${LOAD_CHECKPOINT_PATH} \
       --save ${SAVE_CHECKPOINT_PATH} \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 0 \
       --use-fused-rotary-pos-emb \
       --no-masked-softmax-fusion \
       --no-load-optim \
       --no-load-rng \
       --train-iters 50000 \
       --lr-warmup-fraction 0.01 \
       --mlp-layer-fusion \
       --use-flash-attn \
       --use-fused-rmsnorm \
       --is-instruction-dataset \
       --lora-target-modules query_key_value dense proj dense_4h_to_h \
       --lora-r 16 \
       --lora-alpha 32 \
       --bf16 | tee ./logs/ascendspeed-llama2-7b-ptd-bf16_train-${logfile}.log

# ptd模式训练脚本

