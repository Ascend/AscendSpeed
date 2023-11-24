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
import os
import argparse
import shutil
import torch
import numpy as np
from transformers import LlamaTokenizer, pipeline, LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

#量化权重310P修正
def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    correction = quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset) * deq_scale.npu()
    bias_correction = fp_bias.npu() - correction
    return bias_correction

#cut quant weights
#cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(weight, world_size, cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
                cut_col_keys=['o_proj', 'down_proj']):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in weight.items():
        key_short = key.split('.')[-1]
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list

# cut quant bias
def cut_bias(bias, world_size, is_bias=False, cut_row_keys=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
    state_dict_list = [{} for _ in range(world_size)]
    for key, tensor in bias.items():
        cut_tensor_list = []
        key_short = key.split('.')[-1]
        if key_short in cut_row_keys: 
            # 如果对应的weight做了竖切, bias也做切分
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        else:
            if is_bias:
                 # weight横切，bias除2
                tensor = tensor / 2.0
            cut_tensor_list = [tensor] * world_size
        
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/data/models/llama2-7b-int8",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data/models/llama2-7b-int8_part_model',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default=['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default=['o_proj', 'down_proj'],
        help="cut_col_keys",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size)
    weight_path = args.input_path

    print(f"========= Load Quant Weight =========")
    input_offset_dict = np.load(os.path.join(weight_path, "input_offset.npy"), allow_pickle=True).item()
    quant_weight_dict = np.load(os.path.join(weight_path, "quant_weight.npy"), allow_pickle=True).item()
    fp_bias_dict = np.load(os.path.join(weight_path, "fp_bias.npy"), allow_pickle=True).item()
    deq_scale_dict = np.load(os.path.join(weight_path, "deq_scale.npy"), allow_pickle=True).item()

    print(f"========= Quant Weight BiasCorrection For 301P ==========")
    bias_dict = {}
    for i in fp_bias_dict.keys():
        bias_dict[i] = bias_correction(fp_bias_dict[i], quant_weight_dict[i], int(input_offset_dict[i]), deq_scale_dict[i]).cpu()

    print(f"========= Quant Weight Cut Start ==========")
    state_quant_weight_dict_list = cut_weights(quant_weight_dict, args.world_size, args.cut_row_keys, args.cut_col_keys)
    print(f"========= Quant Bias Cut Start ==========")
    state_bias_dict_list = cut_bias(bias_dict, args.world_size, True, args.cut_row_keys)
    print(f"========= Quant DeqScale Cut Start ==========")
    state_deq_scale_dict_list = cut_bias(deq_scale_dict, args.world_size, False, args.cut_row_keys)

    out_path = args.output_path
    for i in range(args.world_size):
        base_path = os.path.join(out_path, str(i))
        os.makedirs(base_path, exist_ok=True)
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "bias.npy"), state_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])
        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))

    print('save success.')
    print("the location of parallel quant weight is {}".format(out_path))

