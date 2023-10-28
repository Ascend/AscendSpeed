

#  Contents

- [Baichuan-7B](#contents)
  - [Training](#pre-training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
    - [Accuracy of the loss](#accuracy-of-the-loss)

- [Baichuan-13B](#contents)
  - [Training](#pre-training)
  - [Script](#script)
  - [Performance](#performance)
    - [Machine performance](#machine-performance)
    - [Accuracy of the loss](#accuracy-of-the-loss)

# Baichuan-7B

## Training

Here's a hardware summary of pre-training Baichuan-7B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs                   |

Here's a software summary of pre-training Baichuan-7B: 


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
pip install torch_npu-1.11.0.post4_XXXXXX-cp37-cp37m-linux_aarch64.whl

# install megatron-core
pip3 install -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core

# install deepspeed and deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..


# install other packages
pip install -r requirements.txt 
```
3. Prepare pretrained weights
Download the Baichuan-7B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main) 

```shell
  #!/bin/bash
  mkdir tokenizer
  cd ./tokenizer
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/config.json
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/generation_config.json
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/special_tokens_map.json
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenization_baichuan.py
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer.model
  wget https://huggingface.co/baichuan-inc/Baichuan-7B/resolve/main/tokenizer_config.json
  cd ..
```

4. Prepare dataset

Download the Baichuan-7B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
  # download datasets
  mkdir dataset_baichuan
  cd ./dataset_baichuan
  wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  cd ..

  # process datasets                              
  python ./tools/preprocess_data.py \
    --input ./dataset_baichuan/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./tokenizer \
    --output-prefix ./dataset_baichuan/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```


5. Config Baichuan-7B pre-training script : examples/baichuan/pretrain_baichuan_zero_7B.sh 

```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script orign dataset path according to your own dataset path
TOKENIZER_PATH=./tokenizer/  #tokenizer path
DATA_PATH=./dataset_baichuan/alpaca_text_document  #processed dataset
```

6. Launch Baichuan-7B  pre-training script :examples/baichuan/pretrain_baichuan_zero_7B.sh 

```shell
bash examples/baichuan/pretrain_baichuan_zero_7B.sh 
```



### Performance

#### Machine performance

The performance of Baichuan-7B in **Ascend NPU** and **Reference**:

| Device | Model       | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
| ------ | ----------- | ---------------- | ----------------------------- | ---------------------------- | ------------------------- | ----------------------------------- |
| NPUs   | Baichuan-7B | 1024             | 3.250                         | 1914                         | 2.14                      | 102.69                              |
| Reference   | Baichuan-7B | 1024             | 3.978                         | 2068                         | 1.98                      | 125.66                              |



#### Accuracy of the loss

NPU vs Reference loss.

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected. The relative error of the average loss is 0.01093, less than 2%, the maximum relative error is 0.1243, and the maximum absolute error is 0.4859. The precision meets the requirements.

![NPU-LOSS](../../sources/images/baichuan/7B_loss_compare.png)

NPU vs Reference loss relative error.

![NPU-Relative-Error](../../sources/images/baichuan/7B_relative_error.png)



# Baichuan-13B

## Training

Here's a hardware summary of pre-training Baichuan-13B:

| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   NPU    |               8 x Ascend NPUs               |

Here's a software summary of pre-training Baichuan-13B:


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
pip install torch_npu-1.11.0.post4_XXXXXX-cp37-cp37m-linux_aarch64.whl

#install megatron
git clone https://github.com/NVIDIA/Megatron-LM.git -b 23.05
cd Megatron-LM
pip3 install -e ./

# install deepspeed and deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..

# install other packages
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. Prepare pretrained weights


Download the Baichuan-13B checkpoint from [here](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/tree/main) 
```shell
  mkdir tokenizer
  cd ./tokenizer
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/config.json
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/configuration_baichuan.py
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/generation_config.json
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/modeling_baichuan.py
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/pytorch_model-00001-of-00003.bin
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/pytorch_model-00002-of-00003.bin
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/pytorch_model-00003-of-00003.bin
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/pytorch_model.bin.index.json
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/quantizer.py
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/special_tokens_map.json
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/tokenization_baichuan.py
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/tokenizer_config.json
  wget https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/tokenizer.model
  cd ..
```

In order to adapt to the baichuan-13B model, the following script is used to convert the model pre-training weights.
```shell
mkdir model_weights

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir ./tokenizer \
    --output-model-dir ./model_weights \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --make-vocab-size-divisible-by 1 \
    --type 13B \
    --pse True     
```

4. Prepare dataset
Download the Baichuan-13B datasets from [here](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) 

```shell
  mkdir dataset_baichuan
  mkdir model_save
  cd ./dataset_baichuan
  wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  cd ..

```

```shell
#!/bin/bash

python ./tools/preprocess_data.py \
    --input ./dataset_baichuan/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./tokenizer \
    --output-prefix ./dataset_baichuan/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF 
```


5. Config Baichuan-13B pre-training script: /examples/baichuan/pretrain_baichuan_ptd_13B.sh


```shell
# modify the script according to your own  ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# modify script orign dataset path according to your own dataset path
TOKENIZER_PATH=./tokenizer/  
DATA_PATH=./dataset_baichuan/aplaca_text_document  
LOAD_PATH=./model_weights
CHECKPOINT_PATH=./ckpt
```

6. Launch Baichuan-13B pre-training script: /examples/baichuan/pretrain_baichuan_ptd_13B.sh

```bash
bash examples/baichuan/pretrain_baichuan_ptd_13B.sh
```

There is an hourly pulse checking script running that checks that the training is either running or scheduled.



### Performance

#### Machine performance

The performance of the Baichuan-13B in **Ascend NPU** and **Reference**:

| Device |    Model     | total Iterations | throughput rate (samples/s/p) | throughput rate (tokens/s/p) | single-step time (s/step) | floating point operation (TFLOPs/s) |
| :----: | :----------: | :--------------: | :---------------------------: | :--------------------------: | :-----------------------: | :---------------------------------: |
|  NPUs  | Baichuan-13B |       1000       |             1.928             |             1024             |          16.067           |                89.37                |
|  Reference  | Baichuan-13B |       1000       |             1.535             |             862              |          19.852           |                72.39                |



#### Accuracy of the loss

NPU vs Reference loss.

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected. The relative error of the average loss is 0.00725, less than 2%, the maximum relative error is 0.01978, and the maximum absolute error is 0.10811. The precision meets the requirements.

![NPU-LOSS](../../sources/images/baichuan/13B-loss-compare.png)

NPU vs Reference loss relative error.

The relative error between NPU and Reference Loss is less than 0.02 throughout, as expected.

![NPU-Relative-Error](../../sources/images/baichuan/baichuan13B-loss-relative-error.png)
\
\
<font size=1>If the download of the file fails using 'wget' , you can download it manually while ensuring network security.</font>




