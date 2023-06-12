import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu


torch.cuda.init = torch.npu.init
torch.npu.init()
torch.cuda.default_generators = torch_npu.npu.default_generators