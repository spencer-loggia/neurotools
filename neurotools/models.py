import copy
import numpy as np
import torch
from neurotools.modules import Reverb, WeightedConvolution, ElegantWeightedConvolution, ElegantReverb
import networkx as nx


class ElegantReverbNetwork(torch.nn.Module):
    """
    A class intended to replace the old reverb network with more elegance (and better parallelization). It should run
    much faster with a few drawbacks.
    """
    def __init__(self, num_nodes, node_shape: tuple = (1, 3, 64, 64), inject_noise=True,
                 edge_module=ElegantWeightedConvolution, device='cpu', track_activation_history=False, input_nodes=(1,)):
        """

        :param num_nodes:
        :param node_shape:
        :param inject_noise:
        :param edge_module:
        :param device:
        :param track_activation_history:
        :param input_nodes: if input nodes is None, stim inputs to all nodes. Otherwise, mask is set to only project ot input nodes.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.states = torch.zeros(size=(self.num_nodes + 1, node_shape[1], node_shape[2], node_shape[3]), device=device)
        mask = torch.ones((num_nodes + 1, num_nodes + 1), device=device)
        mask[:, 0] = 0
        if input_nodes is not None:
            input_nodes = np.array(input_nodes) + 1 # compensate for added stim node.
            mask[0, np.delete(np.arange(len(mask)), input_nodes)] = 0
        # synaptic module takes n x c x s1 x s2 input and returns output of the same shape.
        self.edge = edge_module(self.num_nodes + 1, node_shape[2], node_shape[3], kernel_size=4, in_channels=node_shape[1],
                                out_channels=node_shape[1], device=device, mask=mask, inject_noise=inject_noise)
        self.inject_noise = inject_noise
        self.sigmoid = torch.nn.Sigmoid()
        if track_activation_history:
            self.past_states = []
        else:
            self.past_states = None
        self.device = device

    def forward(self, x):
        self.states[0] = x
        activ = self.sigmoid(self.states)
        future_state = self.edge(activ).clone()
        self.edge.update(future_state)
        self.states = .9 * future_state + (.1 * self.states.clone())
        if self.past_states is not None:
            self.past_states.append(self.states.clone())

    def detach(self):
        self.edge.detach()
        self.states = torch.zeros_like(self.states)
        self.past_states = []
        return self

    def parameters(self, recurse: bool = True):
        params = self.edge.parameters()
        return params

    def l1(self):
        return torch.sum(torch.abs(self.edge.out_edge))


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



