import torch
from neurotools.modules import Reverb
import networkx as nx


class ReverbNetwork(torch.nn.Module):

    def __init__(self, structure: nx.DiGraph, node_shape: tuple = (1, 3, 64, 64), input_node: int = 0,
                 inject_noise=False, device='cpu'):
        super().__init__()
        self.architecture = structure.copy()
        self.activation = torch.nn.Sigmoid()
        self.device = None
        for node in self.architecture.nodes():
            state = torch.zeros(node_shape)
            self.architecture.nodes[node]['state'] = state
            self.architecture.nodes[node]['shape'] = node_shape
        for u, v in self.architecture.edges():
            u_data = self.architecture.nodes[u]
            v_data = self.architecture.nodes[v]
            fxn = Reverb(u_data['shape'][2], u_data['shape'][3],
                         kernel_size=4, in_channels=u_data['shape'][1],
                         out_channels=v_data['shape'][1], init_plasticity=0.1)
            self.architecture.edges[(u, v)]['operator'] = fxn
        self.input_node = input_node
        self.architecture.add_node(-1, state=torch.zeros(node_shape), shape=node_shape)
        self.architecture.add_edge(-1, self.input_node, operator=Reverb(node_shape[2], node_shape[3],
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
            # hebbian update
            preds = self.architecture.predecessors(v)
            for u in preds:
                self.architecture.edges[(u, v)]['operator'].update(
                    self.activation(self.architecture.nodes[v]['_future_state']))

        # set current state of all nodes to computed future
        for n, data in self.architecture.nodes(data=True):
            if n == -1:
                continue
            self.architecture.nodes[n]['state'] = data['_future_state'].clone()

    def detach(self):
        for u, v, data in self.architecture.edges(data=True):
            self.architecture.edges[(u, v)]['operator'].detach()
        for n, data in self.architecture.nodes(data=True):
            self.architecture.nodes[n]['state'] = torch.zeros(data['shape']).to(self.device)
            self.architecture.nodes[n]['_future_state'] = torch.zeros(data['shape']).to(self.device)

    def to(self, device):
        self.device = device
        self.detach()
        for u, v, data in self.architecture.edges(data=True):
            self.architecture.edges[(u, v)]['operator'].to(device)
        for n, data in self.architecture.nodes(data=True):
            self.architecture.nodes[n]['state'] = self.architecture.nodes[n]['state'].to(device)
            self.architecture.nodes[n]['_future_state'] = self.architecture.nodes[n]['_future_state'].to(device)
