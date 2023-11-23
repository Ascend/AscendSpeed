import sys

import networkx as nx
import torch
import torch_npu

from ascendspeed import print_rank_0
from ascendspeed.core import parallel_state


class GraphSolver:
    def __init__(self):
        self.total_recompute_cost = 0
        self.total_forward_cost = 0
        self.layers_module = None
        self.recompute_policy = {}

    @staticmethod
    def print_recompute_op(graph):
        recompute_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['recompute']:
                recompute_nodes.append(graph.nodes[node]['name'])
        print_rank_0(f'recompute nodes = {recompute_nodes}')

    @staticmethod
    def dg_init(no_recompute_layer):
        dg = nx.DiGraph()
        dg.add_nodes_from([
            (i, {"name": no_recompute_layer[i]['name'],
                 "mem": no_recompute_layer[i]['memory'],
                 "input": no_recompute_layer[i]['input'],
                 "compute": no_recompute_layer[i]['time'],
                 "recompute": False,
                 "status": "no_status"})
            for i in range(len(no_recompute_layer))
        ])
        dg.add_edges_from([
            (i, i + 1) for i in range(len(no_recompute_layer) - 1)
        ])
        return dg

    @staticmethod
    def broadcast_recompute_policy_in_mp(recompute_policy_list):
        recompute_policy_tensor = torch.tensor(recompute_policy_list,
                                               device=torch.device(f'npu:{torch.distributed.get_rank() % 8}'))
        torch.distributed.broadcast(recompute_policy_tensor, src=parallel_state.get_tensor_model_parallel_src_rank(),
                                    group=parallel_state.get_tensor_model_parallel_group())
        return recompute_policy_tensor.cpu().numpy().tolist()

    @staticmethod
    def apply_policy_to_model(recompute_policy_new, full_model):
        if recompute_policy_new[1] == 0:
            for idx, module in enumerate(full_model):
                if idx < recompute_policy_new[0]:
                    module['recompute'] = True
                else:
                    module['recompute'] = False
                    idx_submodule = 0
                    for layer in module['layers']:
                        if recompute_policy_new[idx_submodule + 2] == 1:
                            layer['recompute'] = True
                        idx_submodule += 1
        else:
            for idx, module in enumerate(full_model):
                if idx < recompute_policy_new[0]:
                    module['recompute'] = False
                    idx_submodule = 0
                    for layer in module['layers']:
                        if recompute_policy_new[idx_submodule + 2] == 1:
                            layer['recompute'] = True
                        idx_submodule += 1

    # minimize the number of memory, results in all recompute
    def calculate_cost_mem(self, g: nx.DiGraph, idx):
        subtotal_cost = 0
        subtotal_compute_cost = 0
        cost = (g.nodes[idx]['mem'] if not g.nodes[idx]['recompute'] else g.nodes[idx]['input'])
        compute_cost = (g.nodes[idx]['compute'] if g.nodes[idx]['recompute'] else 0)

        successors = g.successors(idx)
        successor_cnt = 0
        for successor in successors:
            a, b = self.calculate_cost_mem(g, successor)
            subtotal_cost += a
            subtotal_compute_cost += b
            successor_cnt += 1

        return subtotal_cost + cost, subtotal_compute_cost + compute_cost

    # compute the size of peek memory for a given recompute graph
    def calculate_cost_peek(self, g: nx.DiGraph, idx, recompute_mem, chp_mem):
        recompute = g.nodes[idx]['recompute']
        op_mem = g.nodes[idx]['mem']
        op_input = g.nodes[idx]['input']

        if recompute:
            recompute_mem += op_mem
            chp_mem = chp_mem + op_input
        else:
            recompute_mem = 0
            chp_mem += op_mem + op_input

        successors = g.successors(idx)
        successor_cnt = 0

        cur_max_mem = chp_mem + recompute_mem
        global_max_mem = cur_max_mem
        for successor in successors:
            # if another subpath has not been calcuated, we will need to keep the stash
            c = self.calculate_cost_peek(g, successor, recompute_mem, chp_mem)
            # if another subpath has been calculated, shall we keep the output?
            if c > global_max_mem:
                global_max_mem = c
            successor_cnt += 1
        return global_max_mem

    def cal_transformer_memory(self, model_layers):
        s = 0
        if 'layers' in model_layers:
            for layer in model_layers['layers']:
                if str.isdigit(layer['name']):
                    s += layer['memory']
                else:
                    s += self.cal_transformer_memory(layer)
        return s

    def cal_non_transformer_memory(self, model):
        # total memory used
        model_memory = model['layers'][0]['memory']
        model_layers = model['layers'][0]
        transformer_layer_memory = self.cal_transformer_memory(model_layers)
        non_size = model_memory - transformer_layer_memory
        print_rank_0(f"non size {model_memory} {non_size}")
        return non_size

    def dfs_best(self, g, idx, config):
        if idx >= len(config['layers']):
            self.search_recompute_policy(g, config)
            return
        g.nodes[idx]['recompute'] = False
        self.dfs_best(g, idx + 1, config)
        g.nodes[idx]['recompute'] = True
        self.dfs_best(g, idx + 1, config)

    def search_recompute_policy(self, g, config):
        stash_mem_per_layer, recompute_cost = self.calculate_cost_mem(g, 0)
        peek = self.calculate_cost_peek(g, 0, 0, 0)
        for i in range(config['nlayer']):
            # if it is selective
            stash_mem_total = (stash_mem_per_layer * i + config['full_activation'] * (config['nlayer'] - i)) * config['pp']
            if config['static_memory_layer'] + stash_mem_total + peek < config['device_memory']:
                recompute_total = recompute_cost * i  # * config['pp']
                if recompute_total < self.total_recompute_cost:
                    self.total_recompute_cost = recompute_total
                    self.print_recompute_op(g)
                    self.recompute_policy['config'] = 'n_selective'
                    self.recompute_policy['policy'] = g.copy()
                    self.recompute_policy['n'] = i
                    try:
                        print_rank_0(
                            f"recompute policy {i}-selective: {config['static_memory_layer'] / 1024:.1f} GiB + "
                            f"{stash_mem_total / 1024:.1f} GiB + {peek / 1024:.1f} GiB, "
                            f"speed up compared with all recompute"
                            f" {(self.total_forward_cost - recompute_total) / (4 * self.total_forward_cost) * 100:.2f}%")
                    except ZeroDivisionError:
                        print_rank_0("param error. total_forward_cost is 0.")

            # if there are not enough memory
            stash_mem_total = (stash_mem_per_layer * (config['nlayer'] - i) + config['chp_input'] * i) * config['pp']
            if config['static_memory_layer'] + stash_mem_total + peek < config['device_memory']:
                recompute_total = (
                        recompute_cost * (config['nlayer'] - i) + config['chp_time'] * i)
                if recompute_total < self.total_recompute_cost:
                    self.total_recompute_cost = recompute_total
                    self.print_recompute_op(g)
                    self.recompute_policy['config'] = 'n_full'
                    self.recompute_policy['policy'] = g.copy()
                    self.recompute_policy['n'] = i
                    try:
                        print_rank_0(
                            f"recompute policy {i}-full: {config['static_memory_layer'] / 1024:.1f} GiB + "
                            f"{stash_mem_total / 1024:.1f} ({stash_mem_per_layer * (config['nlayer'] - i)} + "
                            f"{config['chp_input'] * i}) GiB + {peek / 1024:.1f} GiB, "
                            f"speed up compared with all recompute "
                            f"{(self.total_forward_cost - recompute_total) / (4 * self.total_forward_cost) * 100:.2f}%")
                    except ZeroDivisionError:
                        print_rank_0("param error. total_forward_cost is 0.")

    def analyse_policy_to_list(self, full_model, recompute_n, recompute_nodes):
        if "config" in self.recompute_policy and self.recompute_policy["config"] != "n_full":
            recompute_policy_list = [int(recompute_n), 1]
        else:
            recompute_policy_list = [int(recompute_n), 0]
        for layer in full_model[0]['layers']:
            if layer["name"] in recompute_nodes:
                recompute_policy_list.append(1)
                continue
            recompute_policy_list.append(0)
        return recompute_policy_list

    def get_layers_module(self, model):
        if "name" in model and model["name"] == "layers":
            self.layers_module = model
            return
        if "layers" not in model:
            return
        for sub_model in model["layers"]:
            self.get_layers_module(sub_model)


def solve_graph(model, pp, device_memory):
    solver = GraphSolver()
    solver.get_layers_module(model)
    solver.total_recompute_cost = sys.maxsize
    # first layer is not recompute
    total_model = solver.layers_module['layers'][0]
    no_recompute_layer = total_model['layers']
    full_chp_per_layer = total_model['input']
    full_chp_time_per_layer = total_model['time']
    full_activation = total_model['memory']

    num_layers = len(solver.layers_module['layers'])
    solver.total_forward_cost = full_chp_time_per_layer * num_layers
    static_memory = model['used_mem'] + solver.cal_non_transformer_memory(model)

    config = {
        'nlayer': num_layers,
        'static_memory_layer': static_memory,
        'pp': pp,
        'device_memory': device_memory,
        'layers': no_recompute_layer,
        'chp_input': full_chp_per_layer,
        'full_activation': full_activation,
        'chp_time': full_chp_time_per_layer
    }
    print_rank_0(
        f"full input {full_chp_per_layer} full time {full_chp_time_per_layer} full activation {full_activation}")
    generate_recompute_policy(solver, config)


def generate_recompute_policy(solver, config):
    num_layers = config["nlayer"]
    full_chp_per_layer = config["chp_input"]
    static_memory = config["static_memory_layer"]
    no_recompute_layer = config["layers"]
    dg = solver.dg_init(no_recompute_layer)
    stash_mem_per_layer, _ = solver.calculate_cost_mem(dg, 0)
    peek = solver.calculate_cost_peek(dg, 0, 0, 0)
    stash_mem_total = stash_mem_per_layer * num_layers
    print_rank_0(
        f"Without recompute: total mem cost: {static_memory / 1024:.1f} GiB + {stash_mem_total / 1024:.1f} GiB + "
        f"{peek / 1024:.1f} GiB, total recompute 0, speed up over all recompute 25%")

    stash_mem_total = full_chp_per_layer * num_layers
    print_rank_0(
        f"With all recompute: total mem cost: {static_memory / 1024:.1f} GiB + {stash_mem_total / 1024:.1f} GiB + "
        f"{peek / 1024:.1f} GiB, total recompute all")

    print_rank_0("With selective recompute:")
    solver.dfs_best(dg, 1, config)
    if 'policy' not in solver.recompute_policy:
        solver.recompute_policy['policy'] = dg
    if 'n' not in solver.recompute_policy:
        solver.recompute_policy['n'] = num_layers
    rg = solver.recompute_policy['policy']
    recompute_nodes = []
    for node in rg.nodes:
        if rg.nodes[node]['recompute']:
            recompute_nodes.append(rg.nodes[node]['name'])
    recompute_n = solver.recompute_policy['n']
    if "config" in solver.recompute_policy:
        print_rank_0(f'recompute nodes = {recompute_nodes}, {recompute_n} {solver.recompute_policy["config"]}')
    full_model = solver.layers_module['layers']

    recompute_policy_new = solver.analyse_policy_to_list(full_model, recompute_n, recompute_nodes)
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        recompute_policy_new = solver.broadcast_recompute_policy_in_mp(recompute_policy_new)
    print_rank_0(f'recompute_policy_new = {recompute_policy_new}')
    solver.apply_policy_to_model(recompute_policy_new, full_model)
