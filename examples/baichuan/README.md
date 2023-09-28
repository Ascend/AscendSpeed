# Baichuan for PyTorch

# 概述

## 简述

- Baichuan7B 是由百川智能开发的一个开源可商用的大规模预训练语言模型。基于 Transformer 结构，在大约 1.2 万亿 tokens 上训练的 70 亿参数模型，支持中英双语，上下文窗口长度为 4096。在标准的中文和英文 benchmark（C-Eval/MMLU）上均取得同尺寸最好的效果。

- Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。Baichuan-13B 有如下几个特点：

  1. 更大尺寸、更多数据：Baichuan-13B 在 Baichuan-7B 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。

  1. 同时开源预训练和对齐模型：预训练模型是适用开发者的『 基座 』，而广大普通用户对有对话功能的对齐模型具有更强的需求。因此本次开源我们同时发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力，开箱即用，几行代码即可简单的部署。

  1. 开源免费可商用：Baichuan-13B 不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。

  ## 模型结构

  整体模型基于Baichuan-7B，为了获得更好的推理性能，Baichuan-13B 使用了 ALiBi 线性偏置技术，相对于 Rotary Embedding 计算量更小，对推理性能有显著提升；与标准的 LLaMA-13B 相比，生成 2000 个 tokens 的平均推理速度 (tokens/s)，实测提升 31.6%：
具体参数和见下表：
  
  | 模型名称     | 隐含层维度 | 层数 | 头数 | 词表大小 | 总参数量       | 训练数据（tokens） | 位置编码 | 最大长度 |
| ------------ | ---------- | ---- | ---- | -------- | -------------- | ------------------ | -------- | -------- |
  | Baichuan-7B  | 4,096      | 32   | 32   | 64,000   | 7,000,559,616  | 1.2万亿            | RoPE     | 4,096    |
  | Baichuan-13B | 5,120      | 40   | 40   | 64,000   | 13,264,901,120 | 1.4万亿            | ALiBi    | 4,096    |


  ​								

# 准备训练环境

## 准备环境

默认配置需要每张卡有60G以上空闲内存。
- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Python版本 | Torch_Version | 三方库依赖版本  |
  | :--------: | :-----------: | :-------------: |
  | Python 3.7 | PyTorch 1.11  | deepspeed 0.9.2 |

- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装相应的驱动、CANN、FRameworkPTAdapter
  当前已测试的版本是HDK 23.0.RC3.B050，CANN 7.0.RC1.B050，5.0.RC3.B050

  **注: Baichuan基于FP16训练，确保获取的CANN支持FP16**

- 创建conda环境

  ```shell
  conda create -n py37 python=3.7
  conda activate py37
  ```

- 安装依赖

  ```shell
  pip3 install -r requirements.txt
  ```

- 安装deepspeed及对应deepspeed_npu插件

  在模型源码包根目录下执行以下命令，安装deepspeed。

  ```shell
  pip3 install deepspeed==0.9.2 
  git clone https://gitee.com/ascend/DeepSpeed.git
  cd DeepSpeed
  pip install ./
  ```

## 准备数据集

1. 获取数据集
   该任务采用开源数据集[alpaca](https://github.com/lm-sys/FastChat/blob/v0.1.10/playground/data/alpaca-data-conversation.json)进行训练。
2. 数据预处理
   下载`alpaca-data-conversation.json`后放在源码包根目录下，运行源码包根目录下的`preprocess_data.py`脚本对原始数据进行预处理。

##  准备预训练权重

​		运行源码包根目录下的`convert_weights_from_huggingface.py`脚本准备预训练权重。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本

   该模型支持单机8卡训练

   注意：

   - 需要确保CANN支持FP16特性
   - 需要将`DATA_PATH`指向经过预处理后的数据
   - 需要将`TOKENIZER_PATH`指向获取的预训练模型文件夹
   - 需要将`CHECKPOINT_PATH`指向一个空的文件夹，该文件夹用来保存模型
   - 需要将`LOAD_PATH`指向转换后的预训练模型权重
   
   ```
    bash examples/baichuan/pretrain_baichuan_zero_7B.sh  
   ```

   模型训练脚本参数说明如下：
   
   ```
    MASTER_ADDR：         指定主节点的地址。
    MASTER_PORT：         指定主节点的端口号。
    NNODES：              指定节点数，设置为1表示只有一个节点。
    NODE_RANK：           指定当前节点的排名，设置为0表示当前节点是第一个节点。
    NPUS_PER_NODE：       指定每个节点上的处理器数。
    WORLD_SIZE：          指定全局的处理器数，即所有节点上的处理器数之和。
    GLOBAL_BATCH：        指定全局的批次大小，即每次训练时一次性处理的样本数。
    MICRO_BATCH：         指定微批次大小，即每个处理器一次性处理的样本数。
   ```
   
   ```
       --DDP-impl                          # 分布式实现方式。
       --tensor-model-parallel-size        # 使用张量并行。
       --pipeline-model-parallel-size      # 使用模型并行。
       --sequence-parallel                 # 使用序列并行。
       --num-layers                        # 模型的层数。
       --hidden-size                       # 模型的隐藏层大小。
       --ffn-hidden-size                   # 模型的前馈神经网络隐藏层大小。
       --num-attention-heads               # 模型的注意力头数。
       --micro-batch-size                  # 每个GPU的微批次大小。
       --global-batch-size                 # 全局批次大小。
       --seq-length                        # 序列长度。
       --max-position-embeddings           # 最大位置嵌入。
       --train-iters                       # 训练迭代次数。
       --save                              # 模型保存路径。
       --load                              # 模型加载路径。
       --data-path                         # 数据路径。
       --split                             # 数据集分割比例。
       --lr                                # 学习率。
       --lr-decay-style                    # 学习率衰减方式。
       --min-lr                            # 最小学习率。
       --initial-loss-scale                # 初始损失缩放。
       --log-interval                      # 日志输出间隔。
       --fp16                              # 使用半精度浮点数进行训练。
   ```
   
   
   

# 版本说明

## 变更

2023.09.27 首次发布

## FAQ

无