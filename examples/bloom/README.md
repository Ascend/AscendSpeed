# Bloom
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/bloom/README_en.md">English</a> </b> 
</p>


# 目录

- [Bloom-7B](#Bloom-7B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
      - [精度](#精度)
  - [微调](#微调)
    - [全参微调](#全参微调)
    - [低参微调](#低参微调)
  - [推理](#推理)
    - [deepspeed_pipeline](#deepspeed_pipeline)
    - [megatron](#megatron)
  - [评估](#评估)
- [Bloom-176B](#Bloom-176B)
  - [训练](#训练)
    - [脚本](#脚本)
    - [性能](#性能)
      - [吞吐](#吞吐)
      - [精度](#精度)
  - [推理](#推理)
    - [deepspeed_pipeline](#deepspeed_pipeline)
    - [megatron](#megatron)
  - [评估](#评估)
- [举例](#举例)

# Bloom-7B

## 训练

Bloom-7B 训练的硬件配置如下:

| **硬件** | **配置**         |
|--------|----------------|
| NPU    | 1x8 Ascend 910 | 

Bloom-7B 训练的软件配置如下:

|            软件             |                                                      版本                                                      |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|
|          python           |                                                    3.8.18                                                    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    2.0.1                                                     |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |

## 脚本

1. 拷贝仓库到本地服务器：

```shell
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
```

2. 搭建环境

```shell
# python3.8
conda create -n bloom7b python=3.8
conda activate bloom7b

# 安装 torch 和 torch_npu
pip install torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl
pip install torch_npu-2.0.1rc1.postxxxxxxxx-cp38-cp38-linux_aarch64.whl


# 安装 megatron-core
pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core

# 安装 deepspeed 和 deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..

# 安装其余包
pip install -r requirements.txt
```

3. 准备预训练权重

首先下载 Bloom-7B 的 [权重](https://huggingface.co/bigscience/bloom-7b1/tree/main)

```shell
mkdir tokenizer
cd tokenizer
wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
...
cd ..
```

将权重从 huggingface 格式转化为 ascendspeed 可以加载的格式

```shell
#!/bin/bash

SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir "your huggingface checkpoint path" \
    --output-model-dir "your ascendspeed checkpoint path" \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --type 7B \
    --deepspeed
```

4. 准备数据集

下载 Bloom-7B 的 [enwiki数据集](https://huggingface.co/datasets/teven/enwiki_100k).

```shell
# 下载数据集
mkdir enwiki_100k_datasets
cd enwiki_100k_datasets
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00000-of-00006-67bcc7d401923db0.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00001-of-00006-6b8562cbb05789a4.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00002-of-00006-62d2b426a93b0912.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00003-of-00006-36c3d6da04c724b6.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00004-of-00006-48bdf99256dcfa5d.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00005-of-00006-bcb3b3af8d7a4140.parquet
cd ..

# 预处理数据
python ./tools/preprocess_data.py \
  --input ./enwiki_100k_datasets/ \
  --tokenizer-name-or-path ./tokenizer \
  --output-prefix ./enwiki_100k_datasets/enwiki-100k \
  --worker 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF
```

5. 配置 Bloom-7B 预训练脚本: examples/bloom/pretrain_bloom_7b1.sh

```shell
# 修改数据集和词表路径
TOKENIZER_NAME_OR_PATH=/home/bloom_data/vocab_file/
DATA_PATH=/home/bloom_data/enwiki_100k/enwiki-100k_text_document
```

6. 启动 Bloom-7B 预训练脚本: examples/bloom/pretrain_bloom_7b1.sh

```shell
bash examples/bloom/pretrain_bloom_7b1.sh
```

## 微调

### 全参微调
执行流程与预训练一致，配置训练权重路径如下：
```shell
# 修改预训练权重路径
CHECKPOINT_PATH='./ckpt'
```

### 低参微调
启动 Bloom-7B 低参微调脚本: examples/bloom/tune_bloom_7b1.sh

```shell
# 修改预训练权重路径
CHECKPOINT_PATH='./ckpt'
# 修改数据集和词表路径
TOKENIZER_NAME_OR_PATH=/home/bloom_data/vocab_file/
DATA_PATH=/home/bloom_data/alpaca/alpaca
```

```shell
bash examples/bloom/tune_bloom_7b1.sh
```

## 性能

### 吞吐

Bloom-7B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比:

| 设备  | 模型       | 迭代数 | 样本吞吐 (samples/p/s) | tokens吞吐 (tokens/p/s) | 单步迭代时间 (s/step) | 浮点计算数 (TFLOPs/s) |
|-----|----------|-----|--------------------|-----------------------|-----------------|------------------|
| NPUs | Bloom-7B | 1000 | 9.779            | 2503                  | 19.63           | 109.85           |
| 参考  | Bloom-7B | 1000 | 9.894              | 2525                  | 19.40           | 111.19           |



### 精度

NPU vs 参考 loss


![7b_lm_loss.png](..%2F..%2Fsources%2Fimages%2Fbloom%2F7b_lm_loss.png)

NPU vs 参考 loss 相对误差

![relative_error.png](..%2F..%2Fsources%2Fimages%2Fbloom%2Frelative_error.png)

## 推理

AscendSpeed 支持 BLOOM 7B 的文本生成推理.

### deepspeed_pipeline
```text
    # 请注意，评估时需要修改一个deepspeed的bug：
    # 将 `<deepspeed-installed-path>/runtime/pipe/engine.py` 文件里的第671行注释掉：
    # self.total_loss += self.loss.detach()
```
```shell
# 修改 model weight 路径和 tokenizer 路径
CHECKPOINT=/home/model/bloom_7B
VOCAB_FILE=/home/bloom_data/vocab_file/
```

```shell
bash ./examples/bloom/generate_bloom_7b_deepspeed_pipeline.sh
```


### megatron

使用 [convert_weights_from_gptmodelpipe_to_gptmodel.sh](../../tools/ckpt_convert/bloom/convert_weights_from_gptmodelpipe_to_gptmodel.sh) 将bloom-7B的权重转换为推理格式

```bash
SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_gptmodelpipe_to_gptmodel_v2.py
python $SCRIPT_PATH \
    --input-model-dir ${INPUT_PATH} \
    --output-model-dir ${OUTPUT_PATH} \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --type 7B
```

配置 Bloom-7B 推理脚本: examples/bloom/generate_bloom_7B_tp8_pp1.sh

```shell
# 修改 model weight 路径和 tokenizer 路径
CHECKPOINT=/home/model/bloom_7B
VOCAB_FILE=/home/bloom_data/vocab_file/
```

```shell
bash ./examples/bloom/generate_bloom_7B_tp8_pp1.sh
```

## 评估 
配置 Bloom-7B 评估脚本: tasks/evaluation/evaluate_bloom_7b1.sh

```shell
# 修改 model weight 路径和 tokenizer 路径和数据集任务路径
CHECKPOINT=/home/model/bloom_7B
VOCAB_FILE=/home/bloom_data/vocab_file/
DATA_PATH="/dataset/boolq/test"
TASK="boolq"
```

```text
    # 请注意，评估时需要修改一个deepspeed的bug：
    # 将 `<deepspeed-installed-path>/runtime/pipe/engine.py` 文件里的第671行注释掉：
    # self.total_loss += self.loss.detach()
```

```shell
bash tasks/evaluation/evaluate_bloom_7b1.sh
```

<table>
  <thead>
    <tr>
      <th>任务</th>
      <th>验证集</th>
      <th>模型</th>
      <th>昇腾值</th>
      <th>社区值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>Test</td>
      <th>bloom 7b</th>
      <td>0.614</td>
      <td>--</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/cais/mmlu">mmlu</a></td>
      <td>Test</td>
      <th>bloom 7b</th>
      <td>0.251</td>
      <td><a href="https://www.hayo.com/article/648ace24409528db3186ef1c">0.254</a></td>
    </tr>
  </tbody>
</table>


# Bloom-176B

## 训练

BLOOM 模型源于: [A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100).

> Scao, Teven Le, et al. "Bloom: A 176b-parameter open-access multilingual language model." arXiv preprint arXiv:2211.05100 (2022).

Bloom-176B 训练的硬件配置:

| **硬件** | **配置**          |
|--------|-----------------|
| NPU    | 12x8 Ascend 910 | 

Bloom-176B 训练的软件配置:
 
|            软件             |                                                      配置                                                      |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|
|          python           |                                                    3.8.0                                                     |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    2.0.1                                                     |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |                                                                                                  |

## 脚本

1. 将仓库拉取到个人服务器

```shell
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
```

2. 搭建环境

```shell
# python3.8
conda create -n bloom176b python=3.8
conda activate bloom176b

# 安装 torch 和 torch_npu
pip install torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl
pip install torch_npu-2.0.1rc1.postxxxxxxxx-cp38-cp38-linux_aarch64.whl
pip install apex-0.1_ascend_xxxxxxxx-cp38-cp38-linux_aarch64.whl

# 安装 megatron-core
pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core

# 安装 deepspeed 和 deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./
cd ..

# 安装其余依赖
pip install -r requirements.txt
```

3. 准备预训练权重

下载 Bloom-176B [权重](https://huggingface.co/bigscience/bloom/tree/main)

```shell
mkdir tokenizer
cd tokenizer
wget https://huggingface.co/bigscience/bloom/resolve/main/special_tokens_map.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json
wget https://huggingface.co/bigscience/bloom/resolve/main/tokenizer_config.json
...
cd ..
```

将权重格式从 huggingface 格式转换为 AscendSpeed 格式：

```shell
#!/bin/bash

SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_huggingface.py
python $SCRIPT_PATH \
    --input-model-dir "your huggingface checkpoint path" \
    --output-model-dir "your ascendspeed checkpoint path" \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 12 \
    --type 176B \
    --deepspeed \
    --partition-layers 6,6,6,6,6,6,6,6,6,6,6,4
# partition-layers 指定的是PP当中每个stage的层数，总和需要等于70
```
4. 准备数据集

下载 Bloom-176B 的 [数据集](https://huggingface.co/datasets/teven/enwiki_100k). 

```shell
# 下载数据集
mkdir enwiki_100k_datasets
cd enwiki_100k_datasets
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00000-of-00006-67bcc7d401923db0.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00001-of-00006-6b8562cbb05789a4.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00002-of-00006-62d2b426a93b0912.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00003-of-00006-36c3d6da04c724b6.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00004-of-00006-48bdf99256dcfa5d.parquet
wget https://huggingface.co/datasets/teven/enwiki_100k/resolve/main/data/train-00005-of-00006-bcb3b3af8d7a4140.parquet
cd ..

# 处理数据集
python ./tools/preprocess_data.py \
  --input ./enwiki_100k_datasets/ \
  --tokenizer-name-or-path ./tokenizer \
  --output-prefix ./enwiki_100k_datasets/enwiki-100k \
  --worker 4 \
  --log-interval 1000 \
  --tokenizer-type PretrainedFromHF
```

5. 配置 Bloom-176B 预训练脚本: examples/bloom/pretrain_bloom_176b.sh

```shell
# 修改 MASTER_ADDR 为主节点 IP，比如, 90.90.2.166
MASTER_ADDR=localhost

# 修改每个节点的节点序号，主节点序号为 0, 其余节点的序号依次增长到集群节点数量-1
NODE_RANK=0

# 修改数据集路径和词表路径
TOKENIZER_NAME_OR_PATH=/home/bloom_data/vocab_file/
DATA_PATH=/home/bloom_data/enwiki_100k/enwiki-100k_text_document
```

6. 启动 Bloom-176B 预训练脚本: examples/bloom/pretrain_bloom_176b.sh

在集群中的每个节点上启动 examples/bloom/pretrain_bloom_176b.sh 脚本

```shell
bash examples/bloom/pretrain_bloom_176b.sh
```

## 性能

### 吞吐

Bloom-176B 在 **昇腾芯片** 和 **参考芯片** 上的性能对比:

| 设备 | 模型         | 总迭代数 | tokens吞吐 (tokens/p/s) |
|----|------------|------|-----------------------|
| NPUs | Bloom-176B | 1000 | 112                   |
| 参考 | Bloom-176B | NA   | 107                   |

### 精度

NPU vs 参考 loss 

![bloom176b_lm_loss_compare](../../sources/images/bloom/bloom176b_lm_loss_compare.PNG)

单节点loss对比

![bloom176b_1node_lm_loss_compare](../../sources/images/bloom/bloom176b_lm_loss_1node_compare.PNG)

## 推理

AscendSpeed 支持 BLOOM 176B的在线文本生成推理
We support AscendSpeed Inference for text generation with BLOOM 176B (deepspeed or megatron).

### deepspeed_pipeline
```text
    # 请注意，评估时需要修改一个deepspeed的bug：
    # 将 `<deepspeed-installed-path>/runtime/pipe/engine.py` 文件里的第671行注释掉：
    # self.total_loss += self.loss.detach()
```
```shell
# # 修改 model weight 路径和 tokenizer 路径
CHECKPOINT=/home/model/bloom_176B
VOCAB_FILE=/home/bloom_data/vocab_file/
```

```shell
bash ./examples/bloom/generate_bloom_176b_deepspeed_pipeline.sh
```

### megatron

使用 [convert_weights_from_gptmodelpipe_to_gptmodel.sh](../../tools/ckpt_convert/bloom/convert_weights_from_gptmodelpipe_to_gptmodel.sh) 脚本将权重转化为推理格式。
推理需要两节点运行，需要我们手工将权重同步到两节点下，0号节点需要 1-37 层权重，1号节点需要 38-74 层权重，执行脚本如下：
```bash
SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_gptmodelpipe_to_gptmodel_v2.py
python $SCRIPT_PATH \
    --input-model-dir ${INPUT_PATH} \
    --output-model-dir ${OUTPUT_PATH} \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --type 176B
```
### 脚本

配置 Bloom-176B 推理脚本: examples/bloom/generate_bloom_176b_2nodes.sh

```shell
# 修改 MASTER_ADDR 为主节点 IP，比如, 90.90.2.166
MASTER_ADDR=localhost

# 修改每个节点的节点序号，主节点序号为 0, 其余节点的序号依次增长到集群节点数量-1
NODE_RANK=0

# 修改数据集路径和词表路径
CHECKPOINT=/home/model/bloom_176B
VOCAB_FILE=/home/bloom_data/vocab_file/
```

```shell
bash ./examples/bloom/generate_bloom_176b_2nodes.sh
```


## 评估 

配置 Bloom-176B 评估脚本: tasks/evaluation/evaluate_bloom_176b.sh

```shell
# 修改 model weight 路径和 tokenizer 路径和数据集任务路径
CHECKPOINT=/home/model/bloom_176B
VOCAB_FILE=/home/bloom_data/vocab_file/
DATA_PATH="/dataset/boolq/test"
TASK="boolq"
```

```text
    # 请注意，评估时需要修改一个deepspeed的bug：
    # 将 `<deepspeed-installed-path>/runtime/pipe/engine.py` 文件里的第671行注释掉：
    # self.total_loss += self.loss.detach()
```

```shell
bash tasks/evaluation/evaluate_bloom_176b.sh
```
<table>
  <thead>
    <tr>
      <th>任务</th>
      <th>验证集</th>
      <th>模型</th>
      <th>昇腾值</th>
      <th>社区值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/boolq">Boolq</a></td>
      <td>Test</td>
      <th>bloom 176b</th>
      <td>0.657</td>
      <td>--</td>
    </tr>
  </tbody>
</table>

# 举例
1. bloom 7b

![bloom_7b_generate.png](..%2F..%2Fsources%2Fimages%2Fbloom_7b_generate.png)
2. bloom 176b

![bloom_176b_generate.png](..%2F..%2Fsources%2Fimages%2Fbloom_176b_generate.png)

# 引用

```
@article{scao2022bloom,
  title={Bloom: A 176b-parameter open-access multilingual language model},
  author={Scao, Teven Le and Fan, Angela and Akiki, Christopher and Pavlick, Ellie and Ili{\'c}, Suzana and Hesslow, Daniel and Castagn{\'e}, Roman and Luccioni, Alexandra Sasha and Yvon, Fran{\c{c}}ois and Gall{\'e}, Matthias and others},
  journal={arXiv preprint arXiv:2211.05100},
  year={2022}
}
```