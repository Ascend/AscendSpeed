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
        self.transformer_module = None
        self.recompute_policy = {}
        self.layers_combination = []
        self.layer_full_recompute_combination = None
        self.layer_without_recompute_combination = None
        self.layer_recompute_one_combination = None

    @staticmethod
    def get_recompute_op(graph):
        recompute_nodes = []
        for node in graph.nodes:
            if graph.nodes[node]['recompute']:
                recompute_nodes.append(graph.nodes[node]['name'])
        return recompute_nodes

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

    def set_recompute_info_to_module(self, module, recompute_nodes, recompute):
        if not recompute:
            module["recompute"] = False
            for layer in module["layers"]:
                layer["recompute"] = False
            return
        if len(recompute_nodes) == 0:
            module["recompute"] = True
            return
        sub_modules = module["layers"]
        recompute_nodes_length = len(recompute_nodes)
        for i in range(recompute_nodes_length):
            if recompute_nodes[i] == self.layer_recompute_one_combination.broadcast_value:
                sub_modules[i]["recompute"] = True
                continue
            sub_modules[i]["recompute"] = False

    def apply_policy_to_model(self, recompute_policy_list):
        full_layers = self.layers_module["layers"]
        if len(recompute_policy_list) == 0:
            return
        idx = 0
        for policy in recompute_policy_list:
            n = policy[0]
            recompute = False
            recompute_nodes = []
            if policy[1] != self.layer_without_recompute_combination.broadcast_value:
                recompute = True
            if policy[1] == self.layer_recompute_one_combination.broadcast_value:
                recompute_nodes = policy[2:]
            for i in range(idx, idx + n):
                self.set_recompute_info_to_module(full_layers[i], recompute_nodes, recompute)
            idx += n

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

    def cal_non_transformer_memory(self, model):
        # total memory used
        model_memory = model['layers'][0]['memory']
        transformer_layer_memory = self.transformer_module['memory']
        non_size = model_memory - transformer_layer_memory
        print_rank_0(f"non size {model_memory} {non_size}")
        return non_size

    def layers_combination_init(self, g, idx, config):
        if idx == 0:
            self.layer_full_recompute_combination = LayerCombination({
                "name": "full_recompute",
                "num": config["nlayer"],
                "memory": config["chp_input"],
                "cost": config["chp_time"],
                "broadcast_value": 0,
                "policy_name": "n_full"
            })
            self.layers_combination.append(self.layer_full_recompute_combination)
            self.layer_without_recompute_combination = LayerCombination({
                "name": "without_recompute",
                "num": config["nlayer"],
                "memory": config["full_activation"],
                "cost": 0,
                "broadcast_value": 2,
                "policy_name": "n_without"
            })
            self.layers_combination.append(self.layer_without_recompute_combination)
        if idx >= len(config['layers']):
            recompute_nodes = self.get_recompute_op(g)
            if len(recompute_nodes) == len(config['layers']) or len(recompute_nodes) == 0:
                return
            stash_mem_per_layer, recompute_cost = self.calculate_cost_mem(g, 0)
            self.layer_recompute_one_combination = LayerCombination({
                "name": ",".join(recompute_nodes),
                "num": config["nlayer"],
                "memory": stash_mem_per_layer,
                "cost": recompute_cost,
                "broadcast_value": 1,
                "policy_name": "n_selective"
            })
            self.layers_combination.append(self.layer_recompute_one_combination)
            return
        g.nodes[idx]['recompute'] = False
        self.layers_combination_init(g, idx + 1, config)
        g.nodes[idx]['recompute'] = True
        self.layers_combination_init(g, idx + 1, config)

    def get_max_goods_value(self, idx, ans, config):
        i, j, k = idx[0], idx[1], idx[2]
        pre_step_ans = ans[i - 1][j - k]
        if k == 0:
            return pre_step_ans

        goods_value = ans[i][j]
        memory = pre_step_ans.memory + k * self.layers_combination[i].memory
        cost = pre_step_ans.cost + k * self.layers_combination[i].cost
        if pre_step_ans.cost == float('inf'):
            cost = k * self.layers_combination[i].cost
        try:
            device_memory = max(config["device_memory"] - config["static_memory_layer"], 0) / config["pp"]
        except ZeroDivisionError:
            device_memory = max(config["device_memory"] - config["static_memory_layer"], 0)
            print_rank_0("[ERROR] pipeline model parallel world size is 0. ")

        if device_memory >= memory and cost <= goods_value.cost:
            goods_value.memory = memory
            goods_value.cost = cost
            goods_value.layer_names.clear()
            if len(pre_step_ans.layer_names) > 0:
                goods_value.layer_names.extend(pre_step_ans.layer_names)
            goods_value.layer_names.extend(self.layers_combination[i].name for _ in range(k))

        return goods_value

    def print_recompute_policy(self, memory, cost, config):
        fmt_str = "With selective recompute:\n"
        for k, v in self.recompute_policy.items():
            if k == self.layer_full_recompute_combination.name:
                policy_name = self.layer_full_recompute_combination.policy_name
            elif k == self.layer_without_recompute_combination.name:
                policy_name = self.layer_without_recompute_combination.policy_name
            else:
                policy_name = self.layer_recompute_one_combination.policy_name
                fmt_str += "recomputeNodes=[{}], ".format(k)
            fmt_str += "{} {}; ".format(v, policy_name)
        all_recompute_cost = len(self.layers_module["layers"]) * self.layer_full_recompute_combination.cost
        try:
            performance = (all_recompute_cost - cost) / (all_recompute_cost * 4)
        except ZeroDivisionError:
            performance = 0
            print_rank_0("[ERROR] all recompute cost is 0. ")
        fmt_str += "\ntotal mem cost: {:.1f} GiB + {:.1f} GiB, speed up compared with all recompute {:.2%}".format(
            config["static_memory_layer"] / 1024, memory * config["pp"] / 1024, performance)
        print_rank_0(fmt_str)

    def get_all_layer_policy(self, combination_num, layer_num, ans, config):
        layer_nodes = [self.layer_full_recompute_combination.name for _ in range(layer_num)]
        memory = layer_num * self.layer_full_recompute_combination.memory
        cost = layer_num * self.layer_full_recompute_combination.cost
        for i in range(layer_num, 0, -1):
            size = layer_num - len(ans[combination_num][i].layer_names)
            if size != layer_num:
                l_nodes = []
                l_nodes.extend(ans[combination_num][i].layer_names)
                # if the policies of all layers are not found, the remaining layers ues all recompute policy.
                l_nodes.extend(self.layer_full_recompute_combination.name for _ in range(size))
                l_memory = ans[combination_num][i].memory + size * self.layer_full_recompute_combination.memory
                l_cost = ans[combination_num][i].cost + size * self.layer_full_recompute_combination.cost
                if l_cost < cost:
                    cost = l_cost
                    memory = l_memory
                    layer_nodes.clear()
                    layer_nodes.extend(l_nodes)

        for nodes in layer_nodes:
            if nodes not in self.recompute_policy.keys():
                self.recompute_policy.update({nodes: 1})
                continue
            self.recompute_policy.update({nodes: self.recompute_policy[nodes] + 1})

        self.print_recompute_policy(memory, cost, config)

    def knapsack_best(self, config):
        combination_num = len(self.layers_combination)
        layer_num = len(self.layers_module["layers"])
        # make combination index id begin for 1.
        self.layers_combination.insert(0, None)
        # init ans
        ans = [[GoodsValue() for _ in range(layer_num + 1)] for _ in range(combination_num + 1)]
        # find max goods value
        for i in range(1, combination_num + 1):
            for j in range(layer_num + 1):
                k = 0
                while k <= self.layers_combination[i].num and k <= j:
                    ans[i][j] = self.get_max_goods_value([i, j, k], ans, config)
                    k += 1
        self.get_all_layer_policy(combination_num, layer_num, ans, config)

    def analyse_policy_to_list(self):
        recompute_policy_list = []
        full_module_layers = self.layers_module["layers"][0]["layers"]
        module_layers_num = len(full_module_layers)
        for nodes_name, v in self.recompute_policy.items():
            nodes_count = [v]
            if nodes_name == self.layer_without_recompute_combination.name:
                broadcast_value = self.layer_without_recompute_combination.broadcast_value
                nodes_count.extend(broadcast_value for _ in range(module_layers_num + 1))
            elif nodes_name == self.layer_full_recompute_combination.name:
                broadcast_value = self.layer_full_recompute_combination.broadcast_value
                nodes_count.extend(broadcast_value for _ in range(module_layers_num + 1))
            else:
                nodes_count.append(self.layer_recompute_one_combination.broadcast_value)
                recompute_nodes = nodes_name.split(",")
                for layer in full_module_layers:
                    if layer["name"] in recompute_nodes:
                        nodes_count.append(self.layer_recompute_one_combination.broadcast_value)
                        continue
                    nodes_count.append(self.layer_without_recompute_combination.broadcast_value)
            recompute_policy_list.append(nodes_count)
        return recompute_policy_list

    def print_list_to_policy(self, recompute_policy_list):
        layer_names = self.layers_module["layers"][0]["layers"]
        module_layers_num = len(layer_names)
        if len(recompute_policy_list) == 0:
            return
        fmt_str = ">> final selective strategy <<\n"
        for policy in recompute_policy_list:
            n = policy[0]
            if policy[1] == self.layer_without_recompute_combination.broadcast_value:
                policy_name = self.layer_without_recompute_combination.policy_name
            elif policy[1] == self.layer_full_recompute_combination.broadcast_value:
                policy_name = self.layer_full_recompute_combination.policy_name
            else:
                policy_name = self.layer_recompute_one_combination.policy_name
                policy = policy[2:]
                nodes = []
                for i in range(module_layers_num):
                    if policy[i] == self.layer_recompute_one_combination.broadcast_value:
                        nodes.append(layer_names[i]["name"])
                fmt_str += "recomputeNodes=[{}], ".format(",".join(nodes))
            fmt_str += "{} {}\n".format(n, policy_name)
        print_rank_0(fmt_str)

    def get_layers_module(self, model):
        if "name" in model and model["name"] == "layers":
            self.layers_module = model
            return True
        if "layers" not in model:
            return False
        has_transformer_layer = False
        for sub_model in model["layers"]:
            has_transformer_layer = (has_transformer_layer or self.get_layers_module(sub_model))
        if has_transformer_layer:
            self.transformer_module = model
        return False


class LayerCombination:
    def __init__(self, config):
        self.name = config["name"]
        self.num = config["num"]
        self.memory = config["memory"]
        self.cost = config["cost"]
        self.broadcast_value = config["broadcast_value"]
        self.policy_name = config["policy_name"]


class GoodsValue:
    def __init__(self):
        self.layer_names = []
        self.memory = 0
        self.cost = float('inf')


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
    solver.layers_combination_init(dg, 0, config)
    solver.knapsack_best(config)
    recompute_policy_new = solver.analyse_policy_to_list()
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        recompute_policy_new = solver.broadcast_recompute_policy_in_mp(recompute_policy_new)
    solver.apply_policy_to_model(recompute_policy_new)
    solver.print_list_to_policy(recompute_policy_new)
