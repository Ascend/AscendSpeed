# make sure the current working directory is in the project root directory
export LD_LIBRARY_PATH=/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tasks/ckpt_convert/llama/convert_weights_when_tp_pp_change.py \
  --input-model-dir /home/weight/llama-7b-tp2-pp4/iter_00_000 \
  --output-model-dir /home/weight/strategy_trans/llama-7b-tp2-pp4 \
  --src-tensor-model-parallel-size 2 \
  --src-pipeline-model-parallel-size 4 \
  --tgt-tensor-model-parallel-size 4 \
  --tgt-pipeline-model-parallel-size 2 \
  --type 7B