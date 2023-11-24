# LLaMA
<p align="left">
        <b><a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/llama2/README.md">简体中文</a></b> |
        <b>English</b> 
</p>

#  Contents

- [LLaMA2-7B](#contents)
  - [Training](#pre-training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
    - [Accuracy of the loss](#accuracy-of-the-loss)
  - [Inference](#Inference-7B)
  - [Evaluation](#Evaluation-7B)

- [LLaMA2-13B](#contents)
  - [Training](#pre-training)
    - [Script](#script)
    - [Performance](#performance)
      - [Machine performance](#machine-performance)
      - [Accuracy of the loss](#accuracy-of-the-loss)
    - [Inference](#Inference)
    - [Evaluation](#Evaluation)



# LLaMA2-7B

## Training

Here's a hardware summary of pre-training  LLaMA2-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

Here's a software summary of pre-training  LLaMA2-7B: 


|         Software          |   Version   |
| :-----------------------: |:-----------:|
|          Python           |   3.7.16    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    1.11.0                                                    |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |

### Script

1. Clone the repository to your local server:
    ```shell
    git clone https://gitee.com/ascend/AscendSpeed.git 
    cd AscendSpeed 
    mkdir logs
    mkdir ckpt
    ```

2. Build environment
   
    ```bash
    # python3.7
    conda create -n test python=3.7
    conda activate test
    
    # install torch and torch_npu
    pip install torch-1.11.0-cp37-cp37m-manylinux2014_aarch64.whl
    pip install torch_npu-1.11.0*-cp37-cp37m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp37-cp37m-linux_aarch64.whl
    
    # install megatron-core
    pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core
    
    # install deepspeed and deepspeed_npu
    pip install deepspeed==0.9.2
    git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
    cd deepspeed_npu
    pip3 install -e ./
    cd ..
    
    # install other packages
    pip install -r requirements.txt 
    ```
     *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*
    
    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False
    
    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLAMA2-7B checkpoint from [here](https://huggingface.co/daryl149/llama-2-7b-hf/tree/main) 
    
    ```shell
      #!/bin/bash
      mkdir -p llama-2-7b-hf
      cd llama-2-7b-hf
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
      wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
      cd ..
    ```
    
   3.1 weight conversion in deepspeed mode
   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-2-7b model  weight conversion in deepspeed as an example.*
    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # convert to deepspeed weights
    python tools/ckpt_convert/llama/convert_weights_from_huggingface.py --input-model-dir llama-2-7b-hf \
                                                                        --output-model-dir ckpt \
                                                                        --tensor-model-parallel-size 1 \
                                                                        --pipeline-model-parallel-size 1 \
                                                                        --type 7B \
                                                                        --deepspeed
    ```
   3.2 weight conversion in ptd mode
   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-2-7b model weight conversion in ptd as an example.*
   ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # convert to ptd weights
    python tools/ckpt_convert/llama/convert_weights_from_huggingface.py --input-model-dir llama-2-7b-hf \
                                                                        --output-model-dir ./llama2-7b-tp8pp1 \
                                                                        --tensor-model-parallel-size 8 \
                                                                        --pipeline-model-parallel-size 1 \
                                                                        --type 7B \
                                                                        --merge-mlp
   ```
4. pre-training

	4.1 Prepare dataset

	Download the LLAMA2-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    
	```shell
	# download datasets
	mkdir dataset_llama2
	cd ./dataset_llama2
	wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
	cd ..
	
	# process datasets                              
	python ./tools/preprocess_data.py \
		 --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
		 --tokenizer-name-or-path ./llama-2-7b-hf \
		 --output-prefix ./dataset_llama2/alpaca \
		 --workers 4 \
		 --log-interval 1000 \
		 --tokenizer-type PretrainedFromHF
	```

	4.2 pre-training using deepspeed mode
	Config LLAMA2-7B pre-training script: examples/llama2/pretrain_llama2_7b_zero_8p.sh 
	```shell
	# modify the script according to your own ascend-toolkit path
	source /usr/local/Ascend/ascend-toolkit/set_env.sh 
	
	# modify script orign dataset path according to your own dataset path
	TOKENIZER_PATH=./llama-2-7b-hf/  #tokenizer path
	DATA_PATH=./dataset_llama2/alpaca_text_document  #processed dataset
	```

	Launch LLAMA2-7B  pre-training script: examples/llama2/pretrain_llama2_7b_zero_8p.sh
	```shell
	bash examples/llama2/pretrain_llama2_7b_zero_8p.sh 
	```
	
	4.3 pre-training using ptd mode
	Config LLAMA2-7B pre-training script: examples/llama2/pretrain_llama2_7b_ptd.sh 
   ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify config according to your own actual situation
    LOAD_CHECKPOINT_PATH="your init model load path"
    SAVE_CHECKPOINT_PATH="your model ckpt save path"
    TOKENIZER_PATH=./llama-2-7b-hf/  #tokenizer path
    DATA_PATH=./dataset_llama2/alpaca_text_document  #processed dataset
   ```

	Launch LLAMA2-7B  pre-training script: examples/llama2/pretrain_llama2_7b_ptd.sh
   
   ```shell
    bash examples/llama2/pretrain_llama2_7b_ptd.sh 
   ```
5. fine-tuning

	5.1 Prepare fine-tuning dataset 
	Download the LLAMA2-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)    
    ```shell
	# download datasets
	mkdir finetune_dataset
	cd ./finetune_dataset
	wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
	cd ..
	
	# process datasets                              
	python ./tools/preprocess_data.py \
		  --input ./dataset_llama2/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
		  --tokenizer-name-or-path ./llama-2-7b-hf \
		  --output-prefix ./finetune_dataset/alpaca \
		  --workers 4 \
		  --log-interval 1000 \
		  --tokenizer-type PretrainedFromHF \
		  --handler-name GeneralInstructionHandler \
		  --append-eod
    ```
   5.2 fine-tuning using deepspeed mode
   5.2.1 Full Parameters Fine-Tuning
   The configuration script for full parameters fine-tuning  is basically the same as that for pretrain_llama2_7b_zero_8p.sh.*The only difference is the data set.*
   ```bash
   DATA_PATH=./finetune_dataset/alpaca
   ```
   5.2.2 Lora Fine-Tuning
   The Lora fine-tuning script is configured by adding the following lora parameters to the pretrain_llama2_7b_zero_8p.sh script:
   ```bash
       --lora-target-modules query_key_value dense gate_proj up_proj down_proj \
       --lora-r 16 \
       --lora-alpha 32 \
   ```
   If the vocabulary is changed, add the following parameters:
   ```bash
     --lora-modules-to-save word_embeddings lm_head.lm_head \
   ```
   The following parameters are added to the resumable training capability of Lora:
   ```bash
       --load ${ORIGIN_CHECKPOINT}  \
       --lora-load ${LORA_CHECKPOINT} \
   ```
   
   
   5.3 fine-tuning using ptd mode
   *The modification method is the same as that in deepspeed mode. For details, see the previous section.*

### Performance

#### Machine performance

The performance of LLaMA2-7B in **Ascend NPU** and **Reference**:

| Device | Model       | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
| :------: | :-----------: | :----------------: | :-----------------------------: | :----------------------------: | :-------------------------: | :-----------------------------------: |
| NPUs   | LLaMA2-7B | 1024             | 5.19                      | 2662                      | 3.08                   | 122.39                         |
| Reference   | LLaMA2-7B | 1024             | 5.63                      | 2884                       | 2.84                   | 131.96                         |


#### Accuracy of the loss

NPU vs Reference loss.

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected. 
The precision meets the requirements.


The relative error ：0.23% < 2%

![NPU-LOSS](../../sources/images/llama2/llama2-7b-tp8pp1mbs4gbs16-cann1115-Megatron-GPU-loss-releative.png)

The absolute error: 0.00326<0.005

![NPU-LOSS and NPU-Absolute-Error](../../sources/images/llama2/llama2-7b-tp8pp1mbs4gbs16-cann1115-Megatron-GPU-loss-absolute.png)


## Inference-7B
Config llama2-7B inference script: examples/llama2/generate_llama2_7b_ptd.sh
```bash
# modify the script according to your own ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
 
# modify script model path and tokenizer path
TOKENIZER_PATH=./llama2-7b-hf/  #tokenizer path
CHECKPOINT=./llama2-7b-tp8pp1  #model path
```
Launch llama2-7B inference script: examples/llama2/generate_llama2_7b_ptd.sh
```bash
bash examples/llama2/generate_llama2_7b_ptd.sh
```
Some inference samples are as follows:
```
===========================================================
1. If you want to quit, please entry one of [q, quit, exit]
2. To create new title, please entry one of [clear, new]
===========================================================

You >> how are you?

AscendSpeed:

I am doing well, thank you for asking! It's always nice to connect with someone new. How about you?</s>

You >> Give three tips for staying healthy.

AscendSpeed:


1. Eat a balanced diet that includes plenty of fruits, vegetables, whole grains, and lean proteins.
2. Exercise regularly, even if it's just a short walk or some light stretching.
3. Get plenty of sleep and aim for 7-8 hours of rest each night.</s>

You >> How can we reduce air pollution?

AscendSpeed:

Air pollution is a major environmental issue that can have serious health and economic consequences. To reduce air pollution, we need to take a multi-faceted approach that includes individual and collective action. Here are some ways to reduce air pollution:
1. Reduce the use of fossil fuels: Fossil fuels are a major source of air pollution, so reducing their use is an effective way to reduce air pollution. This can be done by switching to renewable energy sources such as solar, wind, and geothermal.
2. Use public transportation: Taking public transportation instead of driving can reduce air pollution. This is because public transportation produces fewer emissions than private vehicles.
3. Drive less: Reducing the number of trips taken by car can help reduce air pollution. This can be done by carpooling, biking, or walking when possible.
4. Plant trees: Planting trees can help reduce air pollution by absorbing carbon dioxide and other pollutants. Trees also provide shade and cool the air, which can reduce the need for air conditioning.
5. Use natural air fresheners: Instead

You >> Any more suggestions, please go ahead and list

AscendSpeed:
them.
6. Avoid burning wood or other materials: Burning wood or other materials releases harmful pollutants into the air. Avoid burning materials such as wood, coal, or other materials that release pollutants.
7. Improve indoor air quality: Poor indoor air quality can contribute to air pollution. Improve indoor air quality by using a HEPA air purifier, avoiding smoking and secondhand smoke, and using natural air fresheners.
8. Educate yourself and others: Educating yourself and others about air pollution and its effects can help reduce air pollution. Share information with friends, family, and colleagues about the importance of reducing air pollution.
9. Support government policies: Supporting government policies that promote clean energy and reduce air pollution can help reduce air pollution.
10. Participate in local clean-up initiatives: Participating in local clean-up initiatives can help reduce air pollution by removing pollutants from the environment.
11. Use eco-friendly cleaning products: Using eco-friendly cleaning products can help reduce air pollution. These products are designed to

You >>

```

## Evaluation-7B
We use MMLU benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/cais/mmlu). The evaluation script eval.sh is as follows:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script model path and tokenizer path
TOKENIZER_PATH=./llama2-7b-hf/  #tokenizer path
CHECKPOINT=./llama2-7b-tp8pp1  #model path
# configure task and data path
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6011
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# configure generation parameters 
python -m torch.distributed.launch $DISTRIBUTED_ARGS tasks/evaluation/evaluation.py   \
     --task-data-path $DATA_PATH \
     --task $TASK \
     --seq-length 4096 \
     --max-new-tokens 1 \
     --max-position-embeddings 4096 \
     --rotary-v3-impl \
     --tensor-model-parallel-size 8 \
     --pipeline-model-parallel-size 1  \
     --num-layers 32  \
     --hidden-size 4096  \
     --ffn-hidden-size 11008 \
     --num-attention-heads 32  \
     --mlp-layer-fusion \
     --load ${CHECKPOINT}  \
     --tokenizer-type PretrainedFromHF  \
     --tokenizer-name-or-path $VOCAB_FILE \
     --tokenizer-not-use-fast \
     --fp16  \
     --micro-batch-size 1  \
     --seed 42 | tee logs/eval_mmlu.log
```
start evaluation
```bash
bash tasks/evaluation/eval.sh
```

Evaluation results
```text
                                subject  question_n       acc
0            high_school_macroeconomics         390  0.466667
1                          formal_logic         126  0.253968
2                     international_law         121  0.652893
3                   college_mathematics         100  0.330000
4                      college_medicine         173  0.421965
5                       world_religions         171  0.725146
6                       moral_scenarios         895  0.220112
7                             nutrition         306  0.513072
8                high_school_statistics         216  0.361111
9                      medical_genetics         100  0.490000
10                    college_chemistry         100  0.300000
11              professional_accounting         282  0.361702
12                     professional_law        1534  0.338331
13                        miscellaneous         783  0.698595
14                            sociology         201  0.651741
15                professional_medicine         272  0.496324
16                    logical_fallacies         163  0.552147
17                     public_relations         110  0.563636
18                      college_biology         144  0.506944
19         high_school_european_history         165  0.612121
20                           philosophy         311  0.556270
21                     abstract_algebra         100  0.310000
22               high_school_psychology         545  0.678899
23         high_school_computer_science         100  0.400000
24               elementary_mathematics         378  0.312169
25               high_school_us_history         204  0.617647
26                     machine_learning         112  0.366071
27                            astronomy         152  0.493421
28                         global_facts         100  0.330000
29              high_school_mathematics         270  0.255556
30               electrical_engineering         145  0.496552
31           high_school_microeconomics         238  0.415966
32                      business_ethics         100  0.540000
33             college_computer_science         100  0.400000
34                  high_school_physics         151  0.317881
35                      human_sexuality         131  0.526718
36                      college_physics         102  0.245098
37  high_school_government_and_politics         193  0.720207
38                            marketing         234  0.747863
39                high_school_geography         198  0.601010
40                     security_studies         245  0.555102
41                high_school_chemistry         203  0.418719
42                           management         103  0.699029
43                        jurisprudence         108  0.537037
44                         econometrics         114  0.350877
45                          human_aging         223  0.591928
46                             virology         166  0.403614
47                       moral_disputes         346  0.528902
48                              anatomy         135  0.451852
49              professional_psychology         612  0.498366
50                   conceptual_physics         235  0.455319
51                    computer_security         100  0.560000
52                   clinical_knowledge         265  0.505660
53                    us_foreign_policy         100  0.680000
54                           prehistory         324  0.570988
55            high_school_world_history         237  0.645570
56                  high_school_biology         310  0.535484
57                                total       14042  0.478422
MMLU Running Time:  18266.85981464386
```

# LLaMA2-13B

## Training

Here's a hardware summary of pre-training  LLaMA2-13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

Here's a software summary of pre-training  LLaMA2-13B: 


|         Software          |   Version   |
| :-----------------------: |:-----------:|
|          Python           |   3.7.16    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    1.11.0                                                    |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |

### Script

1. Clone the repository to your local server:
    ```shell
    git clone https://gitee.com/ascend/AscendSpeed.git 
    cd AscendSpeed 
    mkdir logs
    mkdir ckpt
    ```

2. Build environment
   
    ```bash
    # python3.7
    conda create -n test python=3.7
    conda activate test
    
    # install torch and torch_npu
    pip install torch-1.11.0-cp37-cp37m-manylinux2014_aarch64.whl
    pip install torch_npu-1.11.0*-cp37-cp37m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp37-cp37m-linux_aarch64.whl
    
    # install megatron-core
    pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core
    
    # install deepspeed and deepspeed_npu
    pip install deepspeed==0.9.2
    git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
    cd deepspeed_npu
    pip3 install -e ./
    cd ..
    
    # install other packages
    pip install -r requirements.txt 
    ```
     *Note that if you want to train with the weight from huggingface, please run fix a deepspeed loading checkpointing bug by modified `if zero_sd_list is None` as `if zero_sd_list is None or len(zero_sd_list) == 0` in the `_load_zero_checkpoint` function of `<deepspeed-installed-path>/runtime/engine.py`*
    
    ```text
    # original deepspeed/runtime/engine.py, about #Lines2746-2748
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None:
        return False
    
    # modified
    zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
    if zero_sd_list is None or len(zero_sd_list) == 0:
        return False
    ```
3. Prepare pretrained weights and tokenizer
    Download the LLaMA2-13B checkpoint from [here](https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main) 
    
    ```bash
    git lfs install
    git clone https://huggingface.co/NousResearch/Llama-2-13b-hf
    ```
    
   *Note that if you want to use the weight from huggingface, please run the weight conversion script first. The following uses llama-2-13b model  weight conversion as an example.*
    ```bash
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # convert to deepspeed weights
    python tools/ckpt_convert/llama/convert_weights_from_huggingface.py --input-model-dir llama-2-13b-hf \
                                                                        --output-model-dir ckpt \
                                                                        --tensor-model-parallel-size 8 \
                                                                        --pipeline-model-parallel-size 1 \
                                                                        --type 13B \
    ```

4. Prepare dataset
   
    Download the LLaMA2-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 
        
    ```bash
    # dataset：wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    
    cd WORKSPACE
    mkdir alpaca_preprocessed
    python tools/preprocess_data.py --input WORKSPACE/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
                                    --output-prefix WORKSPACE/alpaca_preprocessed/alpaca \
                                    --tokenizer-type PretrainedFromHF \
                                    --tokenizer-name-or-path WORKSPACE/llama-13b-hf \
                                    --tokenizer-not-use-fast \
                                    --handler-name GeneralInstructionHandler
    ```

5. Config LLaMA2-13B pre-training script: examples/llama2/pretrain_llama2_13B_ptd_8p.sh
   
    ```shell
    # modify the script according to your own ascend-toolkit path
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    
    # modify script orign dataset path according to your own dataset path
    TOKENIZER_PATH=./llama-2-13b-hf/  #tokenizer path
    DATA_PATH=WORKSPACE/alpaca_preprocessed/alpaca   #processed dataset
    ```

6. Launch LLaMA2-13B  pre-training script: examples/llama2/pretrain_llama2_13B_ptd_8p.sh
   
    ```shell
    bash examples/llama2/pretrain_llama2_13B_ptd_8p.sh
    ```

### Performance

#### Machine performance

The performance of LLaMA2-13B in **Ascend NPU** and **Reference**:

| Device |   Model    | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
| :------: |:----------:|:----------------:|:-----------------------------:|:----------------------------:|:-------------------------:|:-----------------------------------:|
| NPUs   | LLaMA2-13B |       5000       |             2.868             |           1468.416           |          89.275           |               126.73                |
| Reference   | LLaMA2-13B |        --        |              --               |             1750             |            --             |                 --                  |


#### Accuracy of the loss

NPU vs Reference loss.

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected. 
The precision meets the requirements.The absolute error of the average loss is 0.0011, less than 0.5%. 

![NPU-LOSS](../../sources/images/llama2/llama2_13b_bf16_loss_absolute.png)

## Inference

We support AscendSpeed Inference for text generation with Llama2 13B.
Inference different from pre-training, such as we need to Load pre-training checkpoint and the length of the output samples:

Config Llama2-13B inference script: examples/llama2/generate_llama2_13B_tp8_pp1.sh

```shell
# modify the model weight path and tokenizer path
CHECKPOINT=./llama2-13b-tp8-pp1/
VOCAB_FILE=./llama2-13b-hf/
```

```shell
bash ./examples/llama2/generate_llama2_13B_tp8_pp1.sh
```
Some inference samples are as follows:
![llama2-13B-generate.png](../../sources/images/llama2/llama2-13B-generate.png)


## Evaluation

We use boolq benchmark to evaluate our model. Benchmark Download [here](https://huggingface.co/datasets/boolq).

```shell
    CHECKPOINT=./llama2-13b-tp8-pp1/
    VOCAB_FILE=./llama2-13b-hf/
    # configure task and data path
    DATA_PATH="./boolq/data/test/"
    TASK="boolq"
    # configure generation parameters 
    python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
           --task-data-path $DATA_PATH \
           --task $TASK\
           --seq-length 4096 \
           --max-new-tokens 32 \
           --max-position-embeddings 4096 \
           --tensor-model-parallel-size 8  \
           --pipeline-model-parallel-size 1  \
           --num-layers 40  \
           --hidden-size 5120  \
           --ffn-hidden-size 13824 \
           --load ${CHECKPOINT}  \
           --num-attention-heads 40 \
           --tokenizer-type PretrainedFromHF  \
           --tokenizer-name-or-path $VOCAB_FILE \
           --tokenizer-not-use-fast \
           --fp16  \
           --micro-batch-size 1  \
           --seed 42 | tee logs/train.log
    # start evaluation
    bash tasks/evaluation/eval.sh
```
