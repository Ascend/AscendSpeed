# LLaMA-65B

This directory contains some of the scripts that were used to produce the results in the AscendSpeed. These scripts is to show the example how to run llama-65B in terminal.

LLaMA model is from: [LLaMA: OPen and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1.pdf)

>Touvron, Hugo, et al. "LLaMA: OPen and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971 (2023).

# Contents

- [Contents](#contents)

- [Pre-Training](#pre-training)
  - [Datasets](#datasets)
  
  - [Script](#script)
  
  - [Performance](#performance)
      - [Machine performance](#machine-performance)
      - [Accuracy of the loss](#accuracy-of-the-loss)
  
- [Citation](#citation)


## Pre-Training

LLaMA's model performace is better than GPT3 with less parameters. The 65B LLaMA model is comparable to Google's Chinchilla-70B and Palm-540B.

Here's a quick summary of training llama:
|              |                         |
| :----------: | :---------------------: |
|   Hardware   |       Ascend NPUs       |
|   Software   |       AscendSpeed       |
| Architecture |     GPT3 w/ extras      |
|   Dataset    |       See datasets      |

### Datasets
The model was trained using alpaca datasets.

### Script
1.Clone the repository to your local server
```shell
git clone https://gitee.com/ascend.AscendSpeed.git
cd AscendSpeed
mkdir logs
mkdir ckpt
```
2.Install AscendSpeed requirement environment.
```shell
# python3.7
conda create -n test python=3.7
conda activate test

# install torch and torch_npu
pip install torch-1.11.0-cp37-cp37m-linux_aarch64.whl
pip install torch_npu-1.11.0.post4_XXXXXX-cp37-cp37m-linux_aarch64.whl
pip install apex-0.1_ascend_XXXXXX-cp37-cp37m-linux_aarch64.whl

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
3.Download llama-65b checkpoint
```shell
mkdir tokenizer
cd ./tokenizer

# make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/decapoda-research/llama-65b-hf
cd ..
```
4.In order to adapt to llama-65b model, the following script is used to convert the model pre-training weights
```shell
mkdir model_weights

SCRIPT_PATH=./tools/ckpt_convert/llama/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
      --input-model-dir ./tokenizer \
      --output-model-dir ./model_weights \
      --tensor-model-parallel-size 8 \
      --pipeline-model-parallel-size 4 \
      --type 65B
```

5.Download dataset
```shell
# for llama, dowload alpaca dataset, like
wget http://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.jason

# download tokenizer configs nad (selective) weights from
# http://huggingface.co/decapoda-research/llama-65b-hf
# revise "LLaMATokenizer" as "LLaMTokenizer" in tokenizer_config.json
mkdir dataset
python tools/preprocess_data.py --input alpaca_data.json\
                                --output-prefix dataset/alpaca\
                                --tokenizer-type PretrainedFromHF\
                                --tokenizer-name-or-path llama-65b-hf
                                --tokenizer-not-use-fast
                                --handler-name GeneralInstructionHandler
```

6.Config llama-65B pre-training script : AscendSpeed/examples/llama/pretrain_llama_65B_ptd_32p.sh

```bash
# modify the script according to your own conda and ascend-toolkit path
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HEEL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

# modify script orign dataset path according to your own dataset path
TOKENIZER_PATH=./dataset/llama_tokenizer # line 16
DATA_PATH=./dataset/llama_text_document # line 17
```

7.Launch llama-65B pre-training script : AscendSpeed/examples/llama/pretrain_llama_65B_ptd_32p.sh

```bash
bash examples/llama/pretrain_llama_65B_ptd_32p.sh
```
Config llama-65B pre-training script for multinode (Launch llama-65B pre-training script on each machine):

```shell
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
```
The Training log will look like these:

```Shell
 iteration  3/50000 | consumed samples: 768 | consumed tokens:  1572864 | elapsed time per iteration (ms):  33818.0 | learning rate:    1.406E-07 | gloabl batch size:  256 | lm loss:  1.200820E+01 | loss scale:  1.0 | grad norm:    9.216 | actual seqlen:  2048 | number of skipped
iterations: 0 | number of nan iterations:   0 | samples per second: 7.570 | TFLOPs: 107.09 |
time (ms)
```

### Performance

#### Machine performance

The performance of the NPUs in **Ascend** and Reference:

|  Device   |   Model   |  throughput rate (tokens/s/p) |
|:---------:|:---------:|  :--------------------------: |
| Reference | llama-65B |             260               |
|   NPUs    | llama-65B |             234               |


#### Accuracy of the loss

NPU vs GPU loss.

The NPU runs smoothly, the resource usage is stable, no errors are reported in the middle of the process, the Loss is on a decreasing trend, and the convergence speed is as expected.

![NPU-LOSS](../../sources/images/llama/loss_chart.png)

NPU vs GPU loss relative error.

The relative error between NPU and GPU Loss is less than 0.02 throughout, as expected.

![NPU-Relative-Error](../../sources/images/llama/compare_chart.png)

## Citation

You may also consider original work in your reference:

```shell
@article{Touvron2023llama,
  title={LLaMA: OPen and Efficient Foundation Language Models},
  author={Hugo Touvron*, Thibaut Lavril*, Gautier Izacard*, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal,
  Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave*, Guillaume Lample*},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}}
```