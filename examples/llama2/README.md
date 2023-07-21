# BLOOM

This directory contains some of the scripts that were used to produce the results in the AscendSpeed. These scripts is to show the example how to run BLOOM in terminal.

BLOOM model is from: [A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100)

> Scao, Teven Le, et al. "Bloom: A 176b-parameter open-access multilingual language model." arXiv preprint arXiv:2211.05100 (2022).

# Contents

TODO：需要更新二级 content.
- [Contents](#contents)
- [Evaluation and Tasks](#evaluation-and-tasks)
  - [Dataset](#datastes)
  - [LLama Evaluation](#llama-evaluation)
  - [Zero-shot Task](#zeroshot-task)

## Pre-Training

BLOOM's architecture is very similar to GPT3 with a few added improvements as will be discussed later in this article.

Here's a quick summary of training bloom:

|               |                             |
| :-----        | :-------------              |
| Hardware      | 96 64GB Altas 910B NPUs     |
| Software      | AscendSpeed                 |
| Architecture  | GPT3 w/ extras              |
| Dataset       | xxxxxxxxxx                  |
| Training time | xxxxxxxxxx                  |

### Datasets

TODO: change the context xxxx. Another important feature from Megatron-LM is the efficient data loader. During start up of the initial training each data set is split into samples of the requested sequence length (2048 for BLOOM) and index is created to number each sample. Based on the training parameters the number of epochs for a dataset is calculated and an ordering for that many epochs is created and then shuffled. For example, if a dataset has 10 samples and should be gone through twice, the system first lays out the samples indices in order [0, ..., 9, 0, ..., 9] and then shuffles that order to create the final global order for the dataset. Notice that this means that training will not simply go through the entire dataset and then repeat, it is possible to see the same sample twice before seeing another sample at all, but at the end of training the model will have seen each sample twice. This helps ensure a smooth training curve through the entire training process. These indices, including the offsets into the base dataset of each sample, are saved to a file to avoid recomputing them each time a training process is started. Several of these datasets can then be blended with varying weights into the final data seen by the training process.

- 46 Languages in 1.5TB of deduplicated massively cleaned up text, converted into 350B unique tokens
- Vocabulary size of the model is 250,680 tokens
- For full details please see The BigScience Corpus A 1.6TB Composite Multilingual Dataset

### Script

To launch the environment use ``:

```Shell
source $six_ALL_CCFRWORK/code/tr11-176B-ml/bigscience/train/tr11-176B-ml/start-tr11-176B-ml
```

There is an hourly pulse checking script running that checks that the training is either running or scheduled.

The Training log will look like these:

```Shell
XXXXX
```

### performance

#### machine performance

The performance of the NPUs in XXXXX(configuration) and GPUs is:

TODO：通过表格呈现吞吐性能，还有并行配置

#### Accuracy of the loss

NPU vs GPU loss. XXXX(Explain more).

![NPU-LOSS](./images/7b_lm_loss.png)

NPU vs GPU loss relative error. XXXX(Explain more).

![NPU-Relative-Error](./images/relative_error.png)

## Fine-tune and Evaluation

TODO：提供微调的方式，先加载权重，再微调脚本，跟预训练格式一样；后面需要提供task的验证结果（待开发）。

## Inference

We support AscendSpeed Inference for text generation with BLOOM 7B1.

#### Model weights

Download the BLOOM model checkpoint from [here](TODO: XXXXX), make sure all chunks are downloaded completely, then use the following command to merge them into a single archive file and extract it:

```bash
cat bloom-7b1.tar.part_* > gbloom-7b1.tar
tar xvf bloom-7b1.tar
```

Set `CHECKPOINT_PATH` in `/generate_bloom.sh` to the path of the extracted folder. Since the checkpoint file is large, it is recommended to use the SSD or RAM disk to reduce the checkpoint loading time. Since the checkpoint we distribute is in 8-way tensor parallel, a conversion scripts is also provided if you need to change the tensor parallel dimension.

```bash
TODO: add convert_tp tools.

python tools/convert_tp.py \
    --input-folder <SRC_CKPT_PATH>  \
    --output-folder <DST_CKPT_PATH> \
    --target-tp <TARGET_TP>
```

### Script

We generate text samples using the `generate_bloom` script. Inference different from pre-training, such as we need to Load pre training checkpoint and the length of the output samples:

```shell
bash ./generate_bloom_7b1.sh
```

Alternatively you can also use DeepSpeed from source:

```Shell
TODO: XXXX
```

### Samples

<details>
<summary><b>Left-To-Right Generation Examples</b></summary>
    
#### Example 1
Input: Who is the greatest artist? The greatest artist is ?

Output: TODO: XXXX.

#### Example 2 (Chinese)
Input: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答：?
  
Output: 问题: XXXX.
</details>

All the provided scripts are tested on 8 910B 64GB GPUs for BLOOM 7B1 (fp16). These scripts might not work for other models or a different number of NPUs.

> Note: Sometimes NPUs memory is not freed when inference deployment crashes. You can free this memory by running kill all python in terminal.

## Citation

You may also consider original work in your reference:

@article{scao2022bloom,
  title={Bloom: A 176b-parameter open-access multilingual language model},
  author={Scao, Teven Le and Fan, Angela and Akiki, Christopher and Pavlick, Ellie and Ili{\'c}, Suzana and Hesslow, Daniel and Castagn{\'e}, Roman and Luccioni, Alexandra Sasha and Yvon, Fran{\c{c}}ois and Gall{\'e}, Matthias and others},
  journal={arXiv preprint arXiv:2211.05100},
  year={2022}
}