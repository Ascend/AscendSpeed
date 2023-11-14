# Chinese-LLaMA-Alpaca
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/AscendSpeed/blob/master/examples/alpaca/README_en.md">English</a> </b> 
    </p>
</p>



这里介绍如何快速复现 AscendSpeed 中 [Chinese-LLaMA-Alpaca](https://arxiv.org/abs/2304.08177) 的测试结果

> Cui, Yang, and Yao, et al. "Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca." arXiv preprint arXiv:2304.08177 (2023).



# 目录

- [模型权重](#模型权重)

- [合并权重](#合并权重)

- [微调](#微调)

  - [训练](#训练)
  - [脚本](#脚本)
  - [性能](#性能)

    - [吞吐](#吞吐)
    - [精度](#精度)

- [推理](#推理)
  - [脚本](#脚本)

  - [举例](#举例)

- [引用](#引用)

  

# 模型权重


下载 [原始 LLaMA 模型](https://github.com/facebookresearch/llama) 权重, 并下载 [Chinese-LLaMA-Alpaca 模型的 LoRA](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 权重, 这里的 Lora 权重可以理解为是LLaMA权重的一个”补丁”

# 合并权重
在合并权重前，请先确认机器是否有足够的内存加载模型权重，比如 7B的模型就要求13~15G 内存，同时基于[SHA256](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md)检查权重的完整性以确保合并成功。
原始 LLaMA 文件包括: tokenizer.model, tokenizer_checklist.chk, consolidated.*.pth, params.json等 

#### 步骤 1: [将原始 LLaMA 模型转化为 huggingface 的格式](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#step-1-%E5%B0%86%E5%8E%9F%E7%89%88llama%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E4%B8%BAhf%E6%A0%BC%E5%BC%8F)
请使用 Transformers 提供的 [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) 脚本将 LLAMA 模型权重转化为 `huggingface` 的格式

```
python convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
```

新的 huggingface 模型文件生成在 `--output_dir` 目录下，如下：

```
config.json
generation_config.json
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
pytorch_model.bin.index.json
special_tokens_map.json
tokenizer_config.json
tokenizer.json
tokenizer.model
```

#### 步骤 2: [结合 LoRA 权重生成完整模型权重](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#step-2-%E5%90%88%E5%B9%B6lora%E6%9D%83%E9%87%8D%E7%94%9F%E6%88%90%E5%85%A8%E9%87%8F%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D)

**单个 LoRA 权重合并** (可应用于 Chinese-LLaMA, Chinese-LLaMA-Plus, Chinese-Alpaca).

下载脚本 [merge_llama_with_chinese_lora.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_llama_with_chinese_lora.py), 并执行:
```
python merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --output_type huggingface \
    --output_dir path_to_merged_hf_dir 
```
参数说明:

- `--base_model`： 存放 HF格式 LLaMA 模型和配置文件的目录 (步骤 1 中生成).
- `--lora_model`： 存放 Chinese LLAMA/Alpaca LoRA 解压文件的目录 
- `--output_type`： 明确输出格式，可以是 `pth` or `huggingface`，默认为 `pth`.
- `--output_dir`：明确输出文件保存目录，默认为 `./`.

**多 LoRA 权重合并** (可应用于 Chinese-Alpaca-Plus 和 Chinese-Alpaca-Pro). 

下载脚本 [merge_llama_with_chinese_lora.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_llama_with_chinese_lora.py), 并执行:
```
python merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type huggingface \
    --output_dir path_to_merged_hf_dir 
```

#### 步骤 3: 合并后检查 SHA256

权重合并后请检查 [SHA256](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md)，由于 HF格式的 SHA256 经常发生变化，一般推荐先转化为 `pth` 格式，在确认 SHA256 正确以后，再根据需要转换为 HF 格式。

#### 步骤 4: 将HF的权重格式转化为AscendSpeed格式

如果要使用AscendSpeed的张量和流水并行策略，执行：

```
python tools/ckpt_convert/llama/convert_weights_from_huggingface.py \
    --input-model-dir path_to_merged_hf_dir \
    --output-model-dir path_to_merged_ascendspeed_dir \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --type 7B                                                                    
```
如果要使用AscendSpeed中DeepSpeed的并行策略，执行:

```
python tools/ckpt_convert/llama/convert_weights_from_huggingface.py \
    --input-model-dir path_to_merged_hf_dir \
    --output-model-dir path_to_merged_ascendspeed_dir \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --type 7B  \ 
    --deepspeed                                                                 
```


# 微调
## 训练
Chinese LLaMA Alpaca-13B 微调的硬件配置:

|  硬件 |       配置        |
|:---:|:---------------:|
| CPU | 4xKunPeng920@3.0GHz，64 Core Pre Socket 256CPUS |
| RAM |               32x64 GB DDR4                     |
| NPU | 8 x Ascend NPUs |


Chinese LLaMA Alpaca-13B 微调的软件配置：

|            软件             |                                                      版本                                                      |
|:-------------------------:|:------------------------------------------------------------------------------------------------------------:|
|          Python           |                                                    3.8.18                                                    |
|          driver           | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|         firmware          | [package](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-900-pod-a2-pid-254184911/software) |
|           CANN            |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
| binary arithmetic package |       [package](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)       |
|           torch           |                                                    2.0.1                                                     |
|         torch_npu         |                             [package](https://gitee.com/ascend/pytorch/releases)                             |




## 脚本


1. 拷贝代码仓到本地服务器

```bash
git clone https://gitee.com/ascend/AscendSpeed.git 
cd AscendSpeed 
mkdir logs
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip install torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl
pip install torch_npu-2.0.1rc1.post_XXXXXX-cp38-cp38-linux_aarch64.whl

# 安装 megatron-core
pip3 install --no-use-pep517 -e git+https://github.com/NVIDIA/Megatron-LM.git@23.05#egg=megatron-core

# 安装 deepspeed 和 deepspeed_npu
pip install deepspeed==0.9.2
git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
cd deepspeed_npu
pip3 install -e ./

# 安装其他包
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
3. 准备数据集
```bash
# 对于llama，下载 alpaca 数据集并将其放入 $DATA_PATH, 比如
wget http://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

# 下载 tokenizer 配置 
# https://huggingface.co/yahma/llama-7b-hf/tree/main
# 将 tokenizer_config.json 文件中的 "LLaMATokenizer" 修改为 "LlamaTokenizer" (这是 huggingface 的一个bug)
# 将 tokenizer 文件放在 $TOKENIZER_PATH
mkdir dataset
python tools/preprocess_data.py --input alpaca_data.json \
                                --output-prefix $DATA_PATH \
                                --tokenizer-type PretrainedFromHF \
                                --tokenizer-name-or-path $TOKENIZER_PATH \
                                --tokenizer-not-use-fast \
                                --handler-name GeneralInstructionHandler
```

4. 配置 Chinese-LLaMA-Alpaca 微调脚本 

通过设置 `$MODEL_PATH` 变量区分 7B/13B/33B 参数，比如，当 `$MODEL_PATH` 入参的字符串可以匹配为 `*7b*` 时，脚本便会使用 7B的参数

* 基于torch拉起任务的启动脚本为 : [Chinese-LLaMA-Alpaca-7B/13B/33B](finetune_chinese_llama_alpaca_7_13_33b_tp4_pp2.sh)

```bash
bash examples/alpaca/finetune_chinese_llama_alpaca_7_13_33b_tp4_pp2.sh
```

* 基于deepspeed拉起任务的启动脚本为 : [Chinese-LLaMA-Alpaca-7B/13B/33B](finetune_chinese_llama_alpaca_7_13_33b_tp1_pp1_deepspeed.sh)

```bash
bash examples/alpaca/finetune_chinese_llama_alpaca_7_13_33b_tp1_pp1_deepspeed.sh
```

## 性能

### 吞吐

以下是 Chinese LLaMA Alpaca-13B 在昇腾芯片和参考芯片上的吞吐对比：

|  芯片  |            模型            | 迭代次数 | 样本吞吐 (samples/s/p) | token吞吐 (tokens/s/p) | 单步时间 (s/step) | 浮点计算次数 (TFLOPs/s) |
|:----:|:------------------------:|:----:|:------------------:|:--------------------:|:-------------:|:-----------------:|
| GPUs | Chinese LLaMA Alpaca-13B | 3000 |        5.83        |       1493.73        |     5.48      |      153.91       |
| NPUs | Chinese LLaMA Alpaca-13B | 3000 |        6.08        |       1556.77        |     5.26      |      160.41       |



### 精度

NPU vs GPU loss.

![NPU-LOSS](../../sources/images/alpaca/13b_lm_loss.png)

NPU vs GPU loss 相对误差.


![NPU-Relative-Error](../../sources/images/alpaca/relative_error.png)


## 推理

AscendSpeed 当前支持 Chinese LLaMA Alpaca-13B 的文本生成推理

### 脚本

推理脚本中配置路径参数：[examples/alpaca/generate_alpaca_13B_tp8_pp1.sh](examples/alpaca/generate_alpaca_13B_tp8_pp1.sh)

```shell
# 修改模型权重和tokenizer词表路径
CHECKPOINT=<checkpoint-path>
VOCAB_FILE=<vocabfile-path>
```

```shell
bash examples/alpaca/generate_alpaca_13B_tp8_pp1.sh
```

## 举例
Chinese LLaMA Alpaca-13B:

![alpaca_13b_generate.png](../../sources/images/alpaca/alpaca_13b_generate.png)



# 引用

```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```