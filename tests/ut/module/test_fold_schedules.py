# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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


import os
import timeit
import math
import unittest
from unittest import mock
import torch
import torch_npu
from wrapt_timeout_decorator import timeout
import ascendspeed.core.pipeline_parallel.schedules as schedule
from ascendspeed.core.pipeline_parallel.schedules import \
                    forward_backward_pipelining_with_foldx_aiao
from ascendspeed.core.pipeline_parallel.schedules import \
                    forward_backward_pipelining_with_foldx_fifo


class Handle:

    @staticmethod
    def wait():
        return True

    @staticmethod
    def is_completed():
        return True


class TestFoldSchedule(unittest.TestCase):

    @timeout(1200)
    def test_fold3d(self):

        # Mock global vars
        schedule.get_model_config = mock.Mock(return_value={})
        schedule.get_model_type = mock.Mock(return_value=0)
        schedule.get_num_microbatches = mock.Mock(return_value=9)
        schedule.parallel_state.get_pipeline_model_parallel_world_size = mock.Mock(return_value=3)

        # Mock forward func and backward func
        schedule.forward_step = mock.Mock(return_value=torch.rand(4, 1))
        schedule.backward_step = mock.Mock(return_value=torch.rand(4, 1))

        # Mock async p2p_communication
        handle = Handle()
        schedule.p2p_communication.async_communicate = \
            mock.Mock(return_value=(torch.rand(4, 1), torch.rand(4, 1), [handle]))
        schedule.p2p_communication.recv_gather = \
            mock.Mock(return_value=torch.rand(4, 1))

        # # Mock the first virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=True)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=False)

        model = torch.nn.Linear(4, 1)

        def forward_step_func(data_iterator, model):
            rank = int(os.getenv('RANK', '0'))

            def loss_func(output_tensor):
                return rank, {'loss_reduced': rank}
            return torch.rand(1, 1), loss_func

        def set_input_tensor(input_tensor):
            return True

        def allreduce_gradients(async_op):
            return [Handle()]

        model.set_input_tensor = set_input_tensor
        model.allreduce_gradients = allreduce_gradients

        forward_backward_func = forward_backward_pipelining_with_foldx_aiao
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )

        # Mock the second virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=False)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=False)

        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )

        # Mock the last virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=False)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=True)

        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )

    @timeout(1200)
    def test_foldx(self):

        # Mock global vars
        schedule.get_model_config = mock.Mock(return_value={})
        schedule.get_model_type = mock.Mock(return_value=0)
        schedule.get_num_microbatches = mock.Mock(return_value=9)
        schedule.parallel_state.get_pipeline_model_parallel_world_size = mock.Mock(return_value=3)

        # Mock forward func and backward func
        schedule.forward_step = mock.Mock(return_value=torch.rand(2, 1))
        schedule.backward_step = mock.Mock(return_value=torch.rand(2, 1))

        # Mock async p2p_communication
        handle = Handle()
        schedule.p2p_communication.async_communicate = \
            mock.Mock(return_value=(torch.rand(2, 1), torch.rand(2, 1), [handle]))
        schedule.p2p_communication.recv_gather = \
            mock.Mock(return_value=torch.rand(2, 1))

        # Mock device's status in the virtual pipeline
        schedule.parallel_state.get_pipeline_model_parallel_rank = mock.Mock(return_value=0)

        # Mock the first virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=True)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=False)

        model = torch.nn.Linear(4, 1)

        def forward_step_func(data_iterator, model):
            rank = int(os.getenv('RANK', '0'))

            def loss_func(output_tensor):
                return rank, {'loss_reduced': rank}
            return torch.rand(1, 1), loss_func

        def set_input_tensor(input_tensor):
            return True

        def allreduce_gradients(async_op):
            return [Handle()]

        model.set_input_tensor = set_input_tensor
        model.allreduce_gradients = allreduce_gradients
        forward_backward_func = forward_backward_pipelining_with_foldx_fifo
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )

        # Mock the second virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=False)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=False)

        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )

        # Mock the last virtual pipeline stage
        schedule.parallel_state.is_pipeline_first_stage = mock.Mock(return_value=False)
        schedule.parallel_state.is_pipeline_last_stage = mock.Mock(return_value=True)

        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=range(0, 100),
            model=[model, model],
            num_microbatches=schedule.get_num_microbatches(),
            seq_length=3,
            micro_batch_size=3,  # unused
            decoder_seq_length=3,  # unused
            forward_only=False,
            collect_non_loss_data=False
            )



