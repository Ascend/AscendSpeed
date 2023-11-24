#  LLaMA2-7B&LLaMA2-13B模型-双芯推理指导

- [概述](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#概述)
- [输入输出数据](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#输入输出数据)
- [推理环境准备](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#推理环境准备)
- [快速上手](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#快速上手)
  - [获取源码及依赖](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#获取源码及依赖)
  - [模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#模型推理)
- [模型推理性能](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/foundation_models/LLaMA-1/13b#模型推理性能)

# 概述

LLaMA（Large Language Model Meta AI），由 Meta AI 发布的一个开放且高效的大型基础语，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 参考实现：

  ```
  https://github.com/facebookresearch/llama
  ```

# 输入输出数据

- 输入数据

  | 输入数据       | 大小                               | 数据类型 | 数据排布格式 | 是否必选 |
  | -------------- | ---------------------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN               | INT64    | ND           | 是       |
  | attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32  | ND           | 否       |

- 输出数据

  | 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# 推理环境准备

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本          | 下载链接 |
| -------------- | ------------- | -------- |
| 固件与驱动     | 23.0.RC3.B082 | -        |
| CANN           | 7.0.RC1.B082  | -        |
| Python         | 3.9.11        | -        |
| PytorchAdapter | 1.11.0        | -        |
| 推理引擎       | -             | -        |

**表 2** 推理引擎依赖

| 软件  | 版本要求 |
| ----- | -------- |
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device   |
| ------- | -------- |
| aarch64 | 300I DUO |
| x86     | 300I DUO |

# 快速上手

## 获取源码及依赖

1. 环境部署

- 1.1. 安装HDK

> 先安装firmwire，再安装driver

  1.1.1. 安装firmwire

  安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-310p-npu-firmware_7.0.0.5.242.run |

  ```bash
  # 安装firmwire
  chmod +x Ascend-hdk-310p-npu-firmware_7.0.0.5.242.run
  ./Ascend-hdk-310p-npu-firmware_7.0.0.5.242.run --full
  ```

  1.1.2. 安装driver

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-310p-npu-driver_23.0.rc3.b082_linux-aarch64.run |
| x86     | Ascend-hdk-310p-npu-driver_23.0.rc3.b082_linux-x86-64.run |

  ```bash
  # 根据CPU架构安装对应的 driver
  chmod +x Ascend-hdk-310p-npu-driver_23.0.rc3.b082_*.run
  ./Ascend-hdk-310p-npu-driver_23.0.rc3.b082_*.run --full
  ```

- 1.2. 安装CANN

> 先安装toolkit 再安装kernel

  1.2.1. 安装toolkit

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_7.0.RC1_linux-x86_64.run |

  ```bash
  # 安装toolkit
  chmod +x Ascend-cann-toolkit_7.0.RC1_linux-*.run
  ./Ascend-cann-toolkit_7.0.RC1_linux-*.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
  1.2.2. 安装kernel

  安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-310p_7.0.RC1_linux.run |

  ```bash
  # 安装 kernel
  chmod +x Ascend-cann-kernels-310p_7.0.RC1_linux.run
  ./Ascend-cann-kernels-310p_7.0.RC1_linux.run --install
  ```

- 1.3. 安装PytorchAdapter

> 先安装torch 再安装torch_npu

  1.3.1 安装torch

  安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | torch-1.11.0-cp39-cp39-linux_aarch64.whl |
| x86     | torch-1.11.0+cpu-cp39-cp39-linux_x86_64.whl |

  根据所使用的环境中的python版本，选择torch-1.11.0相应的安装包。

  ```bash
  # 安装torch 1.11.0 的python 3.9 的arm版本为例
  pip install torch-1.11.0-cp39-cp39-linux_aarch64.whl
  ```

  1.3.2 安装torch_npu

  安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v1.11.0_py39.tar.gz |

> 安装选择与torch版本 以及 python版本 一致的torch_npu版本

  ```bash
  # 安装 torch_npu 以torch 1.11.0 的python 3.9的arm版本为例
  tar -zxvf pytorch_v1.11.0_py39.tar.gz
  pip install torch*_aarch64.whl
  ```

## 推理环境准备

> 安装配套软件。安装python依赖。

  ```
  pip3 install -r requirements.txt
  ```

1. 下载LLaMA2-7B/LLaMA2-13B模型权重，放置到自定义`input_dir`

   ```
   https://huggingface.co/NousResearch/Llama-2-13b-hf
   https://huggingface.co/NousResearch/Llama-2-7b-hf
   ```

   

3. 根据版本发布链接，安装加速库 

   | 加速库包名                                            |
   | ----------------------------------------------------- |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
| Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |
   
   具体使用cxx11abi0 还是cxx11abi1 可通过python命令查询
   
   ```python
   import torch

   torch.compiled_with_cxx11_abi()
   ```
   
   若返回True 则使用 cxx11abi1，否则相反。
   
   ```bash
   # 安装
   chmod +x Ascend-cann-atb_7.0.T10_*.run
   ./Ascend-cann-atb_7.0.T10_*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```
   
3. 根据版本发布链接，解压大模型文件

   | 大模型包名                                                   |
   | ------------------------------------------------------------ |
   | Ascend-cann-transformer-llm_abi_0-pta_{pta_version}-aarch64.tar.gz |
   | Ascend-cann-transformer-llm_abi_1-pta_{pta_version}-aarch64.tar.gz |
   | Ascend-cann-transformer-llm_abi_1-pta_{pta_version}-x86_64.tar.gz |
   | Ascend-cann-transformer-llm_abi_1-pta_{pta_version}-x86_64.tar.gz |

    具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装
   # cd {llm_path}
   tar -xzvf Ascend-cann-transformer-llm_abi*.tar.gz
   source set_env.sh
   ```

   > 注： 每次运行前都需要 source CANN， 加速库，大模型

## 模型推理

1. 切分模型权重 **首次跑模型时**，需要先对模型权重进行**切分**，切分方法如下

- 修改代码

  1. 修改`cut_model_and_run_llama.sh`中`input_dir`为真实`input_dir`
  
  2. 修改`cut_model_and_run_llama.sh`中`output_dir`为自定义路径，用于存放切分后的模型权重

- 执行切分

  ```
  bash cut_model_and_run_llama.sh
  # 切分好的模型权重会存放在自定义的output_dir
  ```

2. **执行模型推理** 模型切分完成后，cut_model_and_run_llama.sh会加载`output_idr`下切分好的模型权重（`output_dir/part_model/0`和`output_dir/part_model/1`）进行推理

- 配置可选参数：最大输入输出长度(Optional)
  默认值为2048，可以根据用户需要, 在脚本中手动**配置最大输入输出长度**，把`modeling_llama_parallel.py`脚本中的变量**MAX_SEQ_LENGTH**改为：**期望的最大输入长度 + 最大输出长度**

- 修改配置参数&执行推理
当前支持单case推理和多case推理。
multicase=0时，单case；
multicase=1时，多case；当前多case推理支持用例排列组合，set_case_pair=1时生效。

  ```
  # 双芯模型权重路径
  output_dir="./llama2-7b_parallel"
  # 指定芯片，默认为0,1
  device_id=0

  # 单case生效
  batch_size=1
  seqlen_in=128
  seqlen_out=128
  
  # 多case生效
  # 单case推理(0) or 多case(1)
  multicase=1
  # 多case推理配置参数，默认执行[1,4,8,16,32]的推理
  multi_batch_size=[1,4,8,16,32]
  set_case_pair=0
  # 以下两个变量set_case_pair=0生效，推理默认case，即输入输出分别为[32,64,128,256,512,1024]组合的36组case;
  # 默认输入长度从2^5到2^10
  seqlen_in_range=[5,11]
  # 默认输出长度从2^5到2^10
  seqlen_out_range=[5,11]
  # 以下两个变量set_case_pair=1生效，推理特定case，默认推理(输入长度，输出长度)分别为(256,64),(256,256),(512,512),(1024,1024)4组case;
  seqlen_in_pair=[256,256,512,1024]
  seqlen_out_pair=[64,256,512,1024]
  # LLAMA2-7B or LLAMA2-13B, 为多case推理时输出文件名字的后缀
  model_name="LLAMA2-7B"
  ```
> 单case: 推理[batch_size,seqlen_in,seqlen_out]这个用例；
> 多case: 默认测试batch=1/4/8/16/32，输入32~1024，输出32~1024这些case的性能；当set_case_pair=1时，测试seqlen_in_pair/seqlen_out_pair中的用例排列组合；
> 多case会将性能数据保存在./multibatch_performance_{model_name}_{device_id}.csv，包括case配置、首token、非首token处理时延;
  ```
  bash cut_model_and_run_llama.sh
  ```

  该命令会运行一次简单的推理实例warm up，并启动后续的推理

- 自定义运行可参考`run_llama_parallel_performance.py`

3. **性能优化选项** 
- 开启多stream性能优化，修改cut_model_and_run_llama.sh中，取消下面环境变量的注释，并执行推理
  ```
  export ATB_USE_TILING_COPY_STREAM=1
  ```
# 模型推理性能

| 硬件形态 | 模型 | Batch | 首token(ms)     |非首token(ms)      |
| ------------------ | ------- | --------- | ----- | --------------- |
| Duo双芯(x86) | LLaMA2-7B| 1 | 188 |  65.32  |
| Duo双芯(x86)  | LLaMA2-13B| 1 | 326 | 110.19 |
> Batch=1, 输入长度和输出长度取[32,64,128,256,512,1024], 共36组case取均值
# 模型推理精度

[基于C-EVAL数据集]
| llama Model 5-shot | Average | Avg(Hard) | STEM  | Social Sciences | Humanities | Others |
| ------------------ | ------- | --------- | ----- | --------------- | ---------- | ------ |
| GPU LLaMA2-7B	|34.10|24.34	|	31.16| 44.36	|	34.24	|29.95  |
| NPU LLaMA2-7B (ours)|34.39 | 25.0 |	31.86| 44.0 |	34.63 	| 30.21  |
| GPU LLaMA2-13B            |39.38|23.03	|	33.02| 44.0	|	46.69	|38.28  |
| NPU LLaMA2-13B (ours)     | 39.30 | 23.68| 33.26| 44.0 |	46.69 	| 37.76  |