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

"""GPT zero-shot evaluation."""

import math

import torch

from ascendspeed import get_args
from ascendspeed import print_rank_0, is_last_rank, print_rank_last
from ascendspeed import get_tokenizer
from ascendspeed.core import parallel_state, tensor_parallel
from ascendspeed.checkpointing import load_checkpoint
from ascendspeed.model import GPTModel, LlamaModel
from ascendspeed.training import get_model
from ascendspeed.utils import get_ltor_masks_and_position_ids, unwrap_model
from ascendspeed.p2p_communication import recv_forward, send_forward
from tasks.finetune_utils import build_data_loader
from deepspeed.accelerator import get_accelerator
from .datasets import build_dataset

# These are needed to unwrap the model, would be nice to put these in ascendspeed.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ascendspeed.model import DistributedDataParallel as LocalDDP
from ascendspeed.model import Float16Module

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator


def get_llama_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        print_rank_0('> building llama model ...')
        see_memory_usage(f"Before Building Model", force=True)
        if eval_metric == 'loss':
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        elif eval_metric is None:
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))
       
        args = get_args()
        with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                                remote_device=None if args.remote_device == 'none' else args.remote_device,
                                config_dict_or_path=args.deepspeed_config,
                                enabled=args.zero_stage == 3,
                                mpu=parallel_state):
            model = LlamaModel(
                parallel_output=parallel_output,
                add_pooler=False,
                pre_process=pre_process,
                post_process=post_process
            )
        see_memory_usage(f"After Building Model", force=True)
        return model
       
    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().to(get_accelerator().device_name()).contiguous().byte()
    tokens_ = batch['text'].long().to(get_accelerator().device_name()).contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    input_tensor = recv_forward()

    # Forward pass through the model.
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == 'loss':
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output.contiguous().float(), labels.contiguous())
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == 'accuracy':
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError('forward method for evaluation metric {} '
                                  'is not implemented.'.format(eval_metric))
    return None


def process_forward(batch):
    tokenizer = get_tokenizer()
    tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    input_str_list = []
    answer_str_list = []
    mutil_str_list = []
    label_list = []

    batch_input, bacth_answer, batch_label = batch
    mutil_choice_number = len(bacth_answer)
    bacth_answer = list(zip(*bacth_answer))
    for input_str, answer_str, label in zip(batch_input, bacth_answer, batch_label):
        assert isinstance(answer_str, (tuple, list)) and len(answer_str) > 1
        input_str_list.append(input_str)
        answer_str_list.append(answer_str)
        mutil_str_list += [input_str + answer for answer in answer_str]
        label_list.append(label)
    mutils = tokenizer.tokenizer(mutil_str_list, padding="max_length", return_tensors='pt').input_ids
    no_padding_mutils = tokenizer.tokenizer(mutil_str_list).input_ids
    inputs = tokenizer.tokenizer(input_str_list).input_ids

    answers = [no_padding_mutil[len(inputs[i//mutil_choice_number]):] for i, no_padding_mutil in enumerate(no_padding_mutils)]
    answer_index = [[len(inputs[i//mutil_choice_number]) - 1, len(no_padding_mutils[i]) - 1] for i in range(len(answers))]
    return mutils, inputs, answers, answer_index, label_list, mutil_choice_number


def custom_forwrad(batch, model):
    args = get_args()

    correct = 0
    tokenizer = get_tokenizer()
    tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
    device = torch.npu.current_device()
   
    mutils, _, answers, answer_index, labels, mutil_choice_number = process_forward(batch)
    input_tokens = mutils.long().to(get_accelerator().device_name()).contiguous()
    labels = torch.tensor(labels, device=device)

    input_tensor = recv_forward()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    unwrapped_model.set_input_tensor(input_tensor)

    attention_mask, _, _ = get_ltor_masks_and_position_ids(
        input_tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    outputs = model(
        input_ids=input_tokens,
        attention_mask=attention_mask
    )
    send_forward(outputs)
    outputs = outputs.reshape(-1, mutil_choice_number, outputs.shape[-2], outputs.shape[-1])
    if parallel_state.is_pipeline_last_stage():
        for i, logits in enumerate(outputs):
            preds = []
            for choice in range(mutil_choice_number):
                res = logits[choice, answer_index[i*mutil_choice_number + choice][0]: answer_index[i*mutil_choice_number + choice][1]]
                res = torch.log_softmax(res, dim=-1)
                res = res[torch.arange(res.shape[0], device=device), torch.tensor(answers[i*mutil_choice_number + choice], device=device)].mean()
                preds.append(res)
            pred = torch.argmax(torch.tensor(preds, device=device), -1)
            correct += pred == labels[i]
    return torch.tensor(correct, device=device)



def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()
    micro_batch_size = args.orig_micro_batch_size if hasattr(args, "orig_micro_batch_size") \
        else args.micro_batch_Size
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_last(f'> working on iteration: {iteration}')
                print_rank_last(f"> corrcet percent: {total_output / max(1, (iteration * micro_batch_size))}")
            # Forward evaluation.
            if eval_metric:
                output = forward_step(batch, model, eval_metric)
            else:
                output = custom_forwrad(batch, model)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)
       
        elif eval_metric is None:
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)


def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task == 'WIKITEXT103':
        eval_metric = 'loss'
    elif args.task in ('BoolQ', 'PIQA', 'HellaSwag'):
        eval_metric = None
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))

    # Set up model and load checkpoint.
    model = get_model(get_llama_model_provider(eval_metric))
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Data stuff.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)
    if hasattr(dataset, "class_num"):
        args.orig_micro_batch_size = args.micro_batch_size
        args.micro_batch_size *= dataset.class_num

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric)

    print_rank_0('done :-)')
