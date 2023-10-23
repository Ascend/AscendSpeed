# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""AscendSpeed Module"""
import os
import abc
import json
import logging
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import deepspeed
from deepspeed.accelerator import get_accelerator

import ascendspeed
from ascendspeed import get_args
from ascendspeed.core import parallel_state, tensor_parallel
from ascendspeed.model.lora_utils import is_enable_lora, get_lora_model_classes


_FLOAT_TYPES = (torch.FloatTensor, get_accelerator().FloatTensor)
_HALF_TYPES = (torch.HalfTensor, get_accelerator().HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, config=None, share_embeddings_and_output_weights=True):
        super(MegatronModule, self).__init__()
        self.config = config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_embeddings_and_output_weights:
                raise Exception('shared_embedding_or_output_weight() called for last '
                                'stage, but share_embeddings_and_output_weights is false')
            return self.word_embeddings.weight

    def initialize_word_embeddings(self):
        args = get_args()
        if not self.share_embeddings_and_output_weights:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_embeddings_and_output_weights is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if parallel_state.is_pipeline_last_stage() and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.padded_vocab_size, self.config.hidden_size,
                config=self.config, init_method=self.config.init_method)
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print("WARNING! Distributed processes aren't initialized, so "
                      "word embeddings in the last layer are not initialized. "
                      "If you are just manipulating a model this is fine, but "
                      "this needs to be handled manually. If you are training "
                      "something is definitely wrong.")
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        if parallel_state.is_rank_in_embedding_group():
            torch.distributed.all_reduce(self.shared_embedding_or_output_weight().data,
                                         group=parallel_state.get_embedding_group())

        # Ensure that encoder(first stage) and decoder(split stage) position
        # embeddings have the same initial parameter values
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if parallel_state.is_rank_in_position_embedding_group() and \
                args.pipeline_model_parallel_split_rank is not None:
            self.language_model.embedding.cuda()
            position_embeddings = self.language_model.embedding.position_embeddings
            torch.distributed.all_reduce(position_embeddings.weight.data,
                                         group=parallel_state.get_position_embedding_group())


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        if val is None:
            return val

        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor

    def forward(self, *inputs, **kwargs):
        if parallel_state.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class MegatronModuleForCausalLMABC(torch.nn.Module, abc.ABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self):
        super(MegatronModuleForCausalLMABC, self).__init__()
        self.top_k = 50
        self.top_p = 1.0
        self.do_sample = False
        self.num_beams = 1
        self.temperature = 1.0
        self.max_length = 128
        self.max_new_tokens = 0
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None
        self.num_return_sequences = 1
        self.length_penalty = 1.0
        self.tokenizer = None
        self.recompute = True
        self.detokenize = True
        self.include_input = False
        self.stream = False
        self.return_output_log_probs = False

    @classmethod
    def from_pretrained(
            cls,
            model_provider,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ):
        """
        This is an API for initializing model and loading weight.

        Parameters:
        ----------
        model_provider(`func`):
            Function used to generate model objects which is similar to the training define.
        pretrained_model_name_or_path(`str`, *optional*, defaults to None):
           File path of Model weight in megatron format (TP, PP may be used).
           If it is None, the random initialized weights will be used.
        """

    def generate(self, input_ids=None, **kwargs):
        """
        This is an API for text generation which complies with most huggingface definition.

        - *greedy decoding* if `do_sample=False`
        - *top-k decoding* if `top_k>0`
        - *top-p decoding* if `top_p>0.0`
        - *beam-search decoding* if `num_beams>1`

        Parameters:
        ----------
        input_ids(str | torch.Tensor):
            The text entered by the user, e.g. 'Hello!'
            Or
            The text, which encoded by tokenizer, entered by the user, e.g. [0, 13, 5, ...]
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        top_k (`int`, *optional*, defaults to 0):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token. Optionally,
            use a list to set multiple *end-of-sequence* tokens.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token. Optionally,
            use a list to set multiple *beginning-of-sequence* tokens.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        tokenizer (`obj`, *optional*, defaults to None):
            If you don't want to use the tokenizer initialized by megatron, you can pass it in HF format here.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences. Only activate in beam search mode.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch. Only activate
            in beam search mode.
        recompute (`bool`, *optional*, defaults to True):
            Whether the model not to uses the last result in computing next token.
        detokenize (`bool`, *optional*, defaults to True):
            Whether to detokenize tokens into characters.
        include_input (`bool`, *optional*, defaults to False):
            Whether the output contains the context instruction.
        stream (`bool`, *optional*, defaults to False):
            Whether the output is streamed one by one.
        return_output_log_probs(`bool`, *optional*, defaults to False):
            Whether to return a probability distribution for each token.
            Note that the accumulated probability (i.e. Score) of the whole sentence will be return in beam search mode.
        """
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_length = kwargs.pop("max_length", 128)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 0)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.recompute = kwargs.pop("recompute", True)
        self.detokenize = kwargs.pop("detokenize", True)
        self.include_input = kwargs.pop("include_input", False)
        self.stream = kwargs.pop("stream", False)
        self.return_output_log_probs = kwargs.pop("return_output_log_probs", False)


class MegatronModuleForCausalLM(MegatronModuleForCausalLMABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        from ascendspeed import get_tokenizer
        from ascendspeed.text_generation import greedy_search_or_sampling
        from ascendspeed.text_generation import beam_search

        args = get_args()
        args.max_tokens_to_oom = args.max_tokens_to_oom if hasattr(args, "max_tokens_to_oom") else 4096
        args.inference_batch_times_seqlen_threshold = args.inference_batch_times_seqlen_threshold \
            if hasattr(args, "inference_batch_times_seqlen_threshold") else 4

        self.padded_vocab_size = args.padded_vocab_size
        self.pipeline_size_larger_than_one = args.pipeline_model_parallel_size > 1

        self.tokenizer_ori = get_tokenizer().tokenizer

        # import module to avoid error of circular import
        self.greedy_search_or_sampling = greedy_search_or_sampling
        self.beam_search_in_sampling = beam_search

    @staticmethod
    def _init_deepspeed_inference(model, args):
        ds_config = {
            "fp16": {
                "enabled": True,
            },
            "bf16": {
                "enabled": False,
            },
            "zero_optimization": {
                "stage": 0,
                "reduce_bucket_size": args.hidden_size * args.hidden_size,
            },
            "steps_per_print": 2000,
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False,
        }
        if hasattr(args, "ds_config") and getattr(args, "ds_config"):
            ds_config = args.ds_config
        elif hasattr(args, "deepspeed_config") and getattr(args, "deepspeed_config"):
            with open(args.deepspeed_config, encoding='utf-8', errors='ignore') as f:
                ds_config = json.load(f, strict=False)

            zero_optimization_info = ds_config.get("zero_optimization")
            if zero_optimization_info and zero_optimization_info.get("stage") > 0:
                logging.warning("Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3. "
                                "Transferring to ZeRO-1")
                ds_config["zero_optimization"]["stage"] = 0

        if args.ds_inference:
            logging.warning("ds_inference is not support now, use normal mode instead.")

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            raise ValueError("For now, in DeepSpeed pipeline mode, the pp should not greater than 1 now.\n"
                             "Please set --pipeline-model-parallel-size 1.")

        engine = deepspeed.initialize(
            model=model,
            config_params=ds_config,
            mpu=parallel_state if args.no_pipeline_parallel else None
        )[0]
        engine.module.eval()

        return engine

    @staticmethod
    def _broadcast_tokens(context_tokens, context_length, master_rank):
        if dist.get_world_size() > 1:
            if dist.get_rank() == master_rank:
                context_tokens_tensor = get_accelerator().LongTensor(context_tokens)
                dist.broadcast(context_tokens_tensor, master_rank)
            else:
                context_tokens_tensor = torch.empty(context_length,
                                                    dtype=torch.int64,
                                                    device=torch.device(get_accelerator().device_name()))
                dist.broadcast(context_tokens_tensor, master_rank)
        else:
            context_tokens_tensor = get_accelerator().LongTensor(context_tokens)

        return context_tokens_tensor

    @staticmethod
    def _check_output(output, stream):
        if not stream:
            full_output = None
            for tmp in output:
                full_output = tmp
            return full_output
        else:
            return output

    @staticmethod
    def _ids_check(ids, tokenizer):
        checked_ids = []
        for per_ids in ids:
            if torch.max(per_ids) >= len(tokenizer):
                warning_info = "The output ids exceeds the tokenizer length, "\
                               "the clamp operation is enforced, please check!!"
                logging.warning(warning_info)
                checked_ids.append(torch.clamp(per_ids, min=0, max=len(tokenizer)) - 1)
            else:
                checked_ids.append(per_ids)
        return checked_ids

    @classmethod
    def from_pretrained(
            cls,
            model_provider, pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ) -> MegatronModuleForCausalLMABC:
        from ascendspeed.training import get_model
        from ascendspeed.checkpointing import load_checkpoint
        from ascendspeed.model import DistributedDataParallel as LocalDDP
        from ascendspeed.utils import unwrap_model

        args = get_args()

        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        args.model = get_model(model_provider)

        if pretrained_model_name_or_path:
            args.load = pretrained_model_name_or_path

        if args.deepspeed:
            args.model[0] = cls._init_deepspeed_inference(args.model[0], args)

        if args.load:
            load_checkpoint(args.model, None, None)

        if not args.deepspeed:
            unwrap_classes = (torchDDP, LocalDDP, Float16Module)
            if is_enable_lora():
                unwrap_classes += get_lora_model_classes()
        else:
            unwrap_classes = (torchDDP, LocalDDP, Float16Module, deepspeed.DeepSpeedEngine)

        return unwrap_model(args.model, unwrap_classes)[0]

    def generate(self, input_ids=None, **kwargs):
        args = get_args()

        if not args.deepspeed and parallel_state.get_data_parallel_world_size() > 1:
            raise ValueError("In this inference mode data parallel is forbidden.")

        super().generate(input_ids=input_ids, **kwargs)

        setattr(args, "text_generation_config", {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "temperature": self.temperature,
            "recompute": self.recompute,
            "return_output_log_probs": self.return_output_log_probs,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "greedy": True if not self.do_sample else False
        })

        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)

        # =======================================
        # Tokenize the prompts and broadcasting,
        # so you don't need to pass the prompt on
        # each process.
        # =======================================
        context_length, context_tokens, master_rank = self._tokenize(input_ids)

        # =======================================
        # For parallel we need to send context tokens
        # to other process
        # =======================================
        context_tokens_tensor = self._broadcast_tokens(context_tokens, context_length, master_rank).unsqueeze(0)
        context_tokens = context_tokens_tensor.cpu().numpy().tolist()

        # =======================================
        # Get the streaming tokens generator
        # =======================================
        if self.num_beams > 1:
            token_stream = self.beam_search_in_sampling(
                args.model[0],
                context_tokens,
                beam_size=self.num_beams,
                stop_token=args.eos_id,
                num_return_gen=self.num_return_sequences,
                length_penalty=self.length_penalty
            )
        else:
            token_stream = self.greedy_search_or_sampling(
                args.model[0],
                context_tokens
            )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        output = self._post_processing(token_stream,
                                       context_length,
                                       self.include_input,
                                       self.detokenize,
                                       self.num_beams)
        return self._check_output(output, self.stream)

    def _init_tokenizer(self, args):
        if self.tokenizer is None:
            self.tokenizer = ascendspeed.global_vars.rebuild_tokenizer(
                args, tokenizer=self.tokenizer_ori)

        if self.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.pad_token_id
        if self.eos_token_id is not None:
            self.tokenizer.eos_token_id = self.eos_token_id
        if self.bos_token_id is not None:
            self.tokenizer.bos_token_id = self.bos_token_id

        if self.tokenizer.eos_token_id is not None:
            args.eos_id = self.tokenizer.eos_token_id
            args.eod_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Your tokenizer doesn't include eos_token.")

    def _tokenize(self, input_ids):
        broadcast_rank = torch.zeros(dist.get_world_size(),
                                     dtype=torch.int64,
                                     device=torch.device(get_accelerator().device_name()))

        if input_ids:
            if isinstance(input_ids, str):
                context_tokens = self.tokenizer.encode(input_ids)
            else:
                context_tokens = input_ids

            context_length = len(context_tokens)
            counts = 1
            broadcast_rank[dist.get_rank()] = 1
        else:
            context_tokens = [self.tokenizer.encode("EMPTY TEXT")]
            context_length = 0
            counts = 0

        input_info = [counts, context_length]
        input_info_tensor = get_accelerator().LongTensor(input_info)
        dist.all_reduce(input_info_tensor)
        dist.all_reduce(broadcast_rank)
        counts = input_info_tensor[0].item()
        if counts == 0:
            raise ValueError("Please pass prompt on at least one process.")
        context_length = input_info_tensor[1].item() // counts
        master_rank = torch.nonzero(broadcast_rank)[0, 0]
        return context_length, context_tokens, master_rank

    def _post_processing(self, token_stream, context_length, include_input, detokenize, num_beams):
        for output, _, log_probs in token_stream:
            if not include_input:
                output = [val[context_length:] for val in output]

            if detokenize:
                try:
                    output_checked = self._ids_check(output, self.tokenizer)
                    output = self.tokenizer.batch_decode(output_checked, skip_special_tokens=True)
                except Exception as e:
                    error_info = "Meet errors when trying to decode the tokens. "\
                                 "Please handle it by yourself."
                    logging.error(error_info)
                    logging.error(e)

            output = output[0] if len(output) == 1 else output

            if not self.return_output_log_probs:
                yield output
            else:
                if num_beams == 1:
                    log_probs = [val[context_length:, :] for val in log_probs] if log_probs is not None else None

                yield output, log_probs[0] if len(log_probs) == 1 else log_probs
