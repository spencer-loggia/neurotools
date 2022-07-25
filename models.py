import torch
from torch.multiprocessing import Pool
from modules import ResistiveTensor, Reverb
import networkx as nx


class ReverbNetwork(torch.nn.Module):

    def __init__(self, structure: nx.DiGraph, node_shape: tuple = (1, 3, 64, 64), input_node: int = 0,
                 inject_noise=True):
        super().__init__()
        self.architecture = structure.copy()
        self.activation = torch.nn.Sigmoid()
        for node in self.architecture.nodes():
            state = ResistiveTensor(node_shape, equilibrium=-.1, init_resistivity=.1)
            self.architecture.nodes[node]['state'] = state
            self.architecture.nodes[node]['shape'] = node_shape
        for u, v in self.architecture.edges():
            u_data = self.architecture.nodes[u]
            v_data = self.architecture.nodes[v]
            fxn = Reverb(u_data['shape'][2], u_data['shape'][3],
                         kernel_size=4, in_channels=u_data['shape'][1],
                         out_channels=v_data['shape'][1])
            self.architecture.edges[(u, v)]['operator'] = fxn
        self.input_node = input_node
        self.architecture.add_node(-1, state=torch.zeros(node_shape))
        self.architecture.add_edge(-1, self.input_node, operator=Reverb(node_shape[2], node_shape[3],
                                                                        kernel_size=4, in_channels=node_shape[1],
                                                                        out_channels=node_shape[1]))
        self.inject_noise = True

    def parameters(self, recurse: bool = True):
        parameters = []
        for u, v, data in self.architecture.edges(data=True):
            parameters += list(data['operator'].parameters())
        for n, data in self.architecture.nodes(data=True):
            if n != -1:
                parameters += list(data['state'].parameters())
        return parameters

    def time_step(self):
        nodes = sorted(list(self.architecture.nodes(data=True)))

        # preform natural weight updates
        for v, data in nodes:
            cur_state = data['state']
            if v != -1:
                cur_state = cur_state.data
            if self.inject_noise:
                noise = torch.normal(size=cur_state.shape, mean=0, std=.05)
                cur_state = cur_state + noise
            activation = self.activation(cur_state)
            self.architecture.nodes[v]['activation'] = activation
            for u in self.architecture.predecessors(v):
                self.architecture.edges[(u, v)]['operator'].update(activation)

        # preform forward pass
        for v, data in nodes:
            if v == -1:
                continue
            preds = self.architecture.predecessors(v)
            for u in preds:
                fxn = self.architecture.edges[(u, v)]['operator']
                activation = self.architecture.nodes[u]['activation']
                update = fxn(activation)
                mod = self.architecture.nodes[v]['state'].clone()
                self.architecture.nodes[v]['state'] = mod + update


if __name__=='__main__':
    # test network
    torch.autograd.set_detect_anomaly(True)
    structure = nx.complete_graph(4, create_using=nx.DiGraph)
    revnet = ReverbNetwork(structure, input_node=1, node_shape=(1, 2, 16, 16))
    optimizer = torch.optim.Adam(lr=.1, params=revnet.parameters())
    for i in range(10):
        optimizer.zero_grad()
        revnet.architecture.nodes[-1]['state'] = torch.normal(mean=0, std=.5, size=(1, 2, 16, 16))
        revnet.time_step()
        loss = torch.sum(revnet.architecture.nodes[3]['state'].data.flatten())
        print(loss)
        loss.backward(retain_graph=True)
        optimizer.step()

