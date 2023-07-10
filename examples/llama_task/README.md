# Contents
- [Contents](#contents)
- [Evaluation and Tasks](#evaluation-and-tasks)
  - [Datastes](#datastes)
  - [LLama Evaluation](#llama-evaluation)
  - [Zeroshot Task](#zeroshot-task)
# Evaluation and Tasks

## Datastes
First of all, You must download the evaluation datasets for the [BoolQ](https://storage.googleapis.com/boolq/dev.jsonl), PIQA ([1](https://yonatanbisk.com/piqa/data/valid.jsonl), [2](https://yonatanbisk.com/piqa/data/valid-labels.lst)), [HellaSwag](https://github.com/rowanz/hellaswag/tree/master/data/hellaswag_val.jsonl) tasks.

## LLama Evaluation

We include zeroshot example scripts for llama evaluation on [BoolQ](https://storage.googleapis.com/boolq/dev.jsonl), PIQA ([1](https://yonatanbisk.com/piqa/data/valid.jsonl), [2](https://yonatanbisk.com/piqa/data/valid-labels.lst)), and [HellaSwag](https://github.com/rowanz/hellaswag/tree/master/data/hellaswag_val.jsonl) accuracy.

For example, you can use the following command to run BoolQ zeroshot task on a Llama-7B parameter model.
<pre>
WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TASK="BoolQ"
VALID_DATA=&#60;boolq dev data path&#62;.jsonl

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type PretrainedFromHF \
               --tokenizer-name-or-path ./dataset/llama/  \
               --tokenizer-not-use-fast \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 8 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --log-interval 1 \
               --layernorm-epsilon 1e-6 \
               --fp16 \
               --no-load-optim \
               --no-load-rng
</pre>

## Zeroshot Task


The following table shows the NPU and [LLama Paper](https://arxiv.org/abs/2302.13971) accuracy achieved by the Zeroshot task of the Llama model. 

| Model Size | BoolQ | PIQA | HellaSwag |
| :---: | :---: | :---: | :---: |
| 7B   | 74.7% \| 76.5% | 78.6% \| 79.8% | 73.9% \| 79.8% | 
| 13B  | 79.5% \| 78.1% | 80.4% \| 80.1% | 77.3% \| 80.1% | 
| 33B  | 83.1% \| 83.1% | 81.7% \| 82.3% | 83.0% \| 82.3% |
| 65B  | 85.5% \| 85.3% | 81.2% \| 82.8% | 82.3% \| 82.8% |
