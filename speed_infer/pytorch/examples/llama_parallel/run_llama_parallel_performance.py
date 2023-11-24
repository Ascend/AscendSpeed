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
import time
import argparse
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

def setup_model_parallel(init_args):
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if local_rank == 0:
        torch_npu.npu.set_device(init_args.device)
    elif local_rank == 1:
        torch_npu.npu.set_device(init_args.device + 1)
    torch.manual_seed(1)
    return local_rank, world_size

def trans_data_format(model, config):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
        if not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is 910B, support ND")
    else:
        # if on 910A or 310P chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        if not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is not 910B, support NZ")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)

def init_model(init_args):
    config_path = os.path.join(init_args.load_path, 'part_model', str(local_rank), 'config.json')
    config_init = LlamaConfig.from_pretrained(config_path)
    tokenizer_path = os.path.join(init_args.load_path, 'tokenizer')
    tokenizer_init = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='left')
    if not torch.distributed.get_rank():
        print("load tokenizer success!")

    part_model_path = os.path.join(init_args.load_path, 'part_model', str(local_rank))
    model_init = LlamaForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
    if not torch.distributed.get_rank():
        print("load model success!")

    trans_data_format(model_init, config_init)
    return model_init, tokenizer_init

def warm_up(init_args, warmup_model, warmup_tokenizer, prompt):
    if not torch.distributed.get_rank():
        print("warm-up start")
    inputs_warm_up = warmup_tokenizer(prompt[:init_args.batch], return_tensors="pt", padding='max_length', truncation=True, max_length=10)
    _ = warmup_model.generate(
        inputs_warm_up.input_ids.npu(),
        min_new_tokens=1,
        max_new_tokens=1,
    )
    torch.npu.empty_cache()
    if not torch.distributed.get_rank():
        print("warm-up success!")

def inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=0, model_name="LLAMA2-7B"):

    if not torch.distributed.get_rank():
        print("inference start")
    # tokenize
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    #infer
    with torch.no_grad():
        torch.npu.synchronize()
        first_token_start = time.time()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), min_new_tokens=1, max_new_tokens=1)
        torch.npu.synchronize()
        first_token_end = time.time()
        torch.npu.synchronize()
        generate_start = time.time()
        generate_ids = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=seqlen_out)
        torch.npu.synchronize()
        generate_end = time.time()

    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # time analysis
    time_of_first_token = (first_token_end - first_token_start) * 1000
    time_generate = (generate_end - generate_start) * 1000
    time_tensor = torch.tensor(
    [time_of_first_token, time_generate], device="npu")

    if torch.distributed.get_world_size() >= 2:
        torch.distributed.all_reduce(time_tensor, torch.distributed.ReduceOp.MAX)

    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    time_per_token = ((time_tensor[1] - time_tensor[0]) / (new_tokens - 1)) if new_tokens - 1 else 0
    time_generate = generate_end - generate_start
    if not torch.distributed.get_rank():
        print("\nQ&A results are as follows:")
        for idx, item in enumerate(res):
            print(f"\n[Q&A {idx+1}]\n",item)
        if multi_case:
            file_utils = open(f"multibatch_performance_{model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
            file_utils.write(f"{int(batch)},{seqlen_in},{seqlen_out},{time_generate:.2f},{(time_of_first_token):.2f},{(time_per_token):.2f}\n")
            file_utils.close()
        # time analysis
        print(f"\nBatch: {batch}, Input tokens number: {len(inputs.input_ids[0])}, Output tokens number: {new_tokens}; reserved {torch.npu.memory_reserved(torch_npu.npu.current_device())/1024/1024} MB, allocated {torch.npu.memory_allocated(torch_npu.npu.current_device())/1024/1024} MB")
        print(f"Generated in {time_generate:.2f} s, first token costs {time_of_first_token:.2f} ms, avg token cost {time_per_token:.2f} ms, inference speed is {(batch*new_tokens/time_generate):.2f} tokens/s")

def multi_specified_cases(init_args, infer_model, infer_tokenizer, prompt, batch):
    
    seqlen_in_pair = [int(length) for length in init_args.seqlen_in_pair.strip('[').strip(']').split(",")]
    seqlen_out_pair = [int(length) for length in init_args.seqlen_out_pair.strip('[').strip(']').split(",")]
    assert len(seqlen_in_pair)==len(seqlen_out_pair), "The number of seqlen_in_pair and seqlen_out_pair parameters must be the same. Please check cut_model_and_run_llama.sh"
    
    for seqlen_in, seqlen_out in zip(seqlen_in_pair,seqlen_out_pair):
        inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=init_args.multi_case, model_name=init_args.model_name)
    
    if not torch.distributed.get_rank():
        print("multicase inference success!")

def multi_default_cases(init_args, infer_model, infer_tokenizer, prompt, batch):

    seqlen_in_range = [int(length) for length in init_args.seqlen_in_range.strip('[').strip(']').split(",")]
    seqlen_out_range = [int(length) for length in init_args.seqlen_out_range.strip('[').strip(']').split(",")]

    for i in range(seqlen_in_range[0],seqlen_in_range[1]):
        seqlen_in = 2 ** i
        for j in range(seqlen_out_range[0], seqlen_out_range[1]):
            seqlen_out = 2 ** j
            inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=init_args.multi_case, model_name=init_args.model_name)
    
    if not torch.distributed.get_rank():
        print("multicase inference success!")

def multicase_inference(init_args, infer_model, infer_tokenizer, prompt):
    if not torch.distributed.get_rank():
        print("multicase inference start")

        file_utils = open(f"multibatch_performance_{init_args.model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
        file_utils.write(f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),first_token_time(ms),avg_token_time(ms)\n")
        file_utils.close()

    multi_batch_size = list(map(int,args.multi_batch_size[1:-1].strip().split(",")))

    for batch in multi_batch_size:

        inputs_warm_up = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=10)
        _ = infer_model.generate(inputs_warm_up.input_ids.npu(),
                                        attention_mask=inputs_warm_up.attention_mask.npu(), min_new_tokens=1, max_new_tokens=1)
        torch.npu.empty_cache()

        if init_args.set_case_pair:
            multi_specified_cases(init_args, infer_model, infer_tokenizer, prompt, batch)
        else:
            multi_default_cases(init_args, infer_model, infer_tokenizer, prompt, batch)

        torch.npu.empty_cache()

if __name__ == "__main__":
    # load path
    parser = argparse.ArgumentParser(
        description="load Model weights and run")
    parser.add_argument(
        "--load_path",
        default="/data/models/llama2-7b_parallel",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--device",
        type=int,
        default="0",
        help="Run model on the specified devices",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default="1",
        help="Set batch_size, the maximum value is 32",
    )
    parser.add_argument(
        "--seqlen_in",
        type=int,
        default="128",
        help="Set input sequence length, the value ranges from 32 to 1024 usually",
    )
    parser.add_argument(
        "--seqlen_out",
        type=int,
        default="128",
        help="Set output sequence length, the value ranges from 32 to 1024 usually",
    )
    parser.add_argument(
        "--model_name",
        default="LLAMA2-7B",
        help="Specify output file name, choose in [LLAMA2-7B, LLAMA2-13B]",
    )
    parser.add_argument(
        "--multi_case",
        type=int,
        default=0,
        help="Single case inference or multi case inference",
    )
    parser.add_argument(
        "--set_case_pair",
        type=int,
        default=0,
        help="set specified case_pair if 1",
    )
    parser.add_argument(
        "--multi_batch_size",
        default=[1,4,8,16,32],
        help="run model with given batch_size",
    )
    parser.add_argument(
        "--seqlen_in_range",
        default=[5,11],
        help="[2^5~2^11), input seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_out_range",
        default=[5,11],
        help="[2^5~2^11), output seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_in_pair",
        default=[128,256,512,1024],
        help="specified case",
    )
    parser.add_argument(
        "--seqlen_out_pair",
        default=[128,256,512,1024],
        help="specified case",
    )
    args = parser.parse_args()
    
    # initialize tensor-parallel mode
    local_rank, world_size = setup_model_parallel(args)

    # adapt torch-npu
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    prompt = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
        ]

    # load tokenizer and model
    model, tokenizer = init_model(args)

    # singlecase_inference or multicase_inference
    if args.multi_case:
        multicase_inference(args, model, tokenizer, prompt)
    else:
        # warm up
        warm_up(args, model, tokenizer, prompt)
        inference(model, tokenizer, prompt, args.batch, args.seqlen_in, args.seqlen_out, args.multi_case)