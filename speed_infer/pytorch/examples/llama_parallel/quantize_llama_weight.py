# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch_npu
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM

SEQ_LEN_OUT = 32

# load fp model weight
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/data/llama2-7b',
                                          trust_remote_code=True)
print("load tokenizer success!")
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='/data/llama2-7b',
                                             torch_dtype=torch.float32, trust_remote_code=True).cpu()
print("load model success!")

'''
# for 中文模型
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]
'''

calib_list = ["Where is the capital of China?",
              "Please make a poem:",
              "I want to learn python, how should I learn it?",
              "Please help me write a job report on large model inference optimization:",
              "What are the most worth visiting scenic spots in China?"]

# quantization
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)

from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

quant_config = QuantConfig(w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False,
                           w_hessian=False)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

# save quant weight
calibrator.save('/data/llama2-7b') 
print('Save quant weight success!')
