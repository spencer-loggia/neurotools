import copy

import numpy as np
import torch
from neurotools.modules import Reverb, WeightedConvolution
import networkx as nx


class ReverbNetwork(torch.nn.Module):

    def __init__(self, structure: nx.DiGraph, node_shape: tuple = (1, 3, 64, 64), input_nodes=(0,),
                 inject_noise=False, edge_module=Reverb, device='cpu', track_activation_history=False):
        super().__init__()
        self.architecture = structure.copy()
        self.activation = torch.nn.Sigmoid()
        self.tracking = track_activation_history
        self.edge_module = edge_module
        self.device = None
        for node in self.architecture.nodes():
            state = torch.zeros(node_shape)
            self.architecture.nodes[node]['state'] = state
            self.architecture.nodes[node]['shape'] = node_shape
            self.architecture.nodes[node]['_past_states'] = []
        for u, v in self.architecture.edges():
            u_data = self.architecture.nodes[u]
            v_data = self.architecture.nodes[v]
            fxn = edge_module(u_data['shape'][2], u_data['shape'][3],
                              kernel_size=4, in_channels=u_data['shape'][1],
                              out_channels=v_data['shape'][1], init_plasticity=0.1)
            self.architecture.edges[(u, v)]['operator'] = fxn
        self.input_nodes = input_nodes
        self.architecture.add_node(-1, state=torch.zeros(node_shape), shape=node_shape)
        for innode in self.input_nodes:
            self.architecture.add_edge(-1, innode, operator=edge_module(node_shape[2], node_shape[3],
                                                                                 kernel_size=4, in_channels=node_shape[1],
                                                                                 out_channels=node_shape[1],
                                                                                 init_plasticity=0.1))
        self.inject_noise = inject_noise

        self.to(device)

    def parameters(self, recurse: bool = True):
        parameters = []
        for u, v, data in self.architecture.edges(data=True):
            parameters += list(data['operator'].parameters())
        return parameters

    def forward(self):
        # preform forward pass
        for v, data in self.architecture.nodes(data=True):
            if v == -1:
                continue
            self.architecture.nodes[v]['_future_state'] = torch.zeros(data['shape']).to(self.device)
            preds = self.architecture.predecessors(v)
            pred_count = 0
            for u in preds:
                fxn = self.architecture.edges[(u, v)]['operator']
                state = self.architecture.nodes[u]['state']
                activation = self.activation(state)
                update = fxn(activation)
                self.architecture.nodes[v]['_future_state'] = self.architecture.nodes[v]['_future_state'] + update
                pred_count += 1
            self.architecture.nodes[v]['_future_state'] = self.architecture.nodes[v]['_future_state'] / pred_count
            # updates.
            preds = self.architecture.predecessors(v)
            for u in preds:
                self.architecture.edges[(u, v)]['operator'].update(
                    self.activation(self.architecture.nodes[v]['_future_state']))

        # set current state of all nodes to computed future
        for n, data in self.architecture.nodes(data=True):
            if n == -1:
                continue
            elif self.tracking:
                self.architecture.nodes[n]['_past_states'].append(data['state'].clone())
            self.architecture.nodes[n]['state'] = data['_future_state'].clone()

    def detach(self):
        for u, v, data in self.architecture.edges(data=True):
            self.architecture.edges[(u, v)]['operator'].detach()
        for n, data in self.architecture.nodes(data=True):
            self.architecture.nodes[n]['state'] = torch.zeros(data['shape']).to(self.device)
            self.architecture.nodes[n]['_future_state'] = torch.zeros(data['shape']).to(self.device)
            if self.tracking:
                self.architecture.nodes[n]['_past_states'] = []
        return self

    def to(self, device):
        self.device = device
        self.detach()
        for u, v, data in self.architecture.edges(data=True):
            self.architecture.edges[(u, v)]['operator'].to(device)
        for n, data in self.architecture.nodes(data=True):
            self.architecture.nodes[n]['state'] = self.architecture.nodes[n]['state'].to(device)
            self.architecture.nodes[n]['_future_state'] = self.architecture.nodes[n]['_future_state'].to(device)
        return self

    def mutate(self):
        weights = []
        edges_ids = []
        for u, v, data in self.architecture.edges(data=True):
            if u == -1:
                continue
            if self.architecture.out_degree(u) == 1 or self.architecture.in_degree(v) == 1:
                # don't want any node to have no out or in edges.
                continue
            weight = data['operator'].get_weight().detach().cpu().item()
            weights.append(weight)
            edges_ids.append((u, v))
        weights = torch.abs(torch.Tensor(weights))
        max_weight = torch.max(weights)
        weights = weights / max_weight
        removal_probs = torch.pow(abs(weights - 1), 3) / 2
        for i, edge in enumerate(edges_ids):
            roll = torch.rand(size=(1,))
            if roll < removal_probs[i]:
                self.architecture.remove_edge(edge[0], edge[1])
                print("removed edge (", edge[0], ",", edge[1], " with roll of", roll.item(),
                      "against chance of", removal_probs[i].item())
        # add edge modifier
        add_prob = 1 / (len(self.architecture.nodes)**3)
        for u, u_data in self.architecture.nodes(data=True):
            for v, v_data in self.architecture.nodes(data=True):
                roll = torch.rand(size=(1,))
                if roll < add_prob:
                    fxn = self.edge_module(u_data['shape'][2], u_data['shape'][3],
                                           kernel_size=4, in_channels=u_data['shape'][1],
                                           out_channels=v_data['shape'][1], init_plasticity=0.1, device=self.device)
                    self.architecture.add_edge(u, v, operator=fxn)
                    print("inserted edge (", u, ",", v, " with roll of", roll.item(),
                          "against chance of", add_prob)
        return self

    def clone(self):
        self.detach()
        new_net = copy.deepcopy(self)
        return new_net


class ElegantReverbNet:
    """
    A class intended to replace the old reverb network with more elegance (and better parallelization)
    """
    def __init__(self, structure: nx.DiGraph, node_shape: tuple = (1, 3, 64, 64), input_nodes=(0,),
                 inject_noise=False, edge_module=Reverb, device='cpu', track_activation_history=False):
        self.num_nodes = len(structure)
        self.states = torch.normal(size=(self.num_nodes, node_shape[1], node_shape[2], node_shape[3]), mean=0, std=.2)
        # synaptic module takes n x c x s1 x s2 input and returns output of the same shape.
        #
