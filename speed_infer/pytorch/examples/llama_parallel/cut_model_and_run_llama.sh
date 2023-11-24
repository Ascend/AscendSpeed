#!/bin/bash
export HCCL_OP_BASE_FFTS_MODE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=110
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
export ATB_CONTEXT_WORKSPACE_RING=1
# export ATB_USE_TILING_COPY_STREAM=1

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

# origin model path
input_dir="./llama2-7b"
# origin parameters
world_size_=2
cut_row_keys_=['q_proj','k_proj','v_proj','gate_proj','up_proj']
cut_col_keys_=['o_proj','down_proj']

# customized parameters
output_dir="./llama2-7b_parallel"
device_id=0
batch_size=1
seqlen_in=128
seqlen_out=128

# single case inference(0) or multi case inference(1)
multi_case=0
# LLAMA2-7B or LLAMA2-13B, use as file name when running multi case inference
model_name="LLAMA2-7B"
multi_batch_size=[1]
seqlen_in_range=[5,11]
seqlen_out_range=[5,11]
set_case_pair=0
seqlen_in_pair=[256,256,512,1024]
seqlen_out_pair=[64,256,512,1024]

script_dir=$(cd $(dirname $0); pwd)
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')

# if model has already been cutted, then run the model; if not, cut the model first
if test -d "$output_dir";
then
    echo "Weight directory exists, runing......"
    cp $script_dir/modeling_llama_parallel.py $transformers_package_path/models/llama/modeling_llama.py
    torchrun --nproc_per_node 2 --master_port 25641 run_llama_parallel_performance.py \
    --load_path $output_dir \
    --device $device_id \
    --batch $batch_size \
    --seqlen_in $seqlen_in \
    --seqlen_out $seqlen_out \
    --multi_case $multi_case \
    --model_name $model_name \
    --multi_batch_size $multi_batch_size \
    --set_case_pair $set_case_pair \
    --seqlen_in_range $seqlen_in_range \
    --seqlen_out_range $seqlen_out_range \
    --seqlen_in_pair $seqlen_in_pair \
    --seqlen_out_pair $seqlen_out_pair
else 
    echo "Cutted Weight directory does not exist, cuting the weight......"
    cp $script_dir/modeling_llama_parallel_cut.py $transformers_package_path/models/llama/modeling_llama.py
    python ./cut_model_util.py \
    --input_path $input_dir \
    --output_path $output_dir \
    --world_size $world_size_ \
    --cut_row_keys $cut_row_keys_ \
    --cut_col_keys $cut_col_keys_
fi
