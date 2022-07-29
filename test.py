import models
import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import networkx as nx
from matplotlib import pyplot as plt

batch_size_train = 1
batch_size_test = 1
structure = nx.complete_graph(2, create_using=nx.DiGraph)
# for node in structure.nodes():
#     structure.add_edge(node, node)
revnet = models.ReverbNetwork(structure, input_node=0, node_shape=(1, 2, 28, 28))
train_loader = torch.utils.data.DataLoader(
    MNIST('./tmp/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    MNIST('./tmp/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

revnet_decoder = torch.nn.Sequential(torch.nn.MaxPool2d(2),
                                     torch.nn.Conv2d(kernel_size=7, in_channels=1, out_channels=10))
present_frames = 10
optimizer = torch.optim.Adam(lr=.01, params=list(revnet_decoder.parameters()) + list(revnet.parameters()))
ce_loss = torch.nn.NLLLoss()
history = []
for epoch in range(10):
    for i, (stim, target) in enumerate(train_loader):
        optimizer.zero_grad()
        revnet.detach()
        if i > 100:
            break
        for i in range(present_frames):
            revnet.architecture.nodes[-1]['state'][:, 0, :, :] = stim
            revnet()
        decode_input = revnet.architecture.nodes[0]['state'][:, 0, :, :][:, None, :, :].clone()
        y_hat = revnet_decoder(decode_input)
        y_hat = y_hat.view(1, 10)
        y_hat = torch.log_softmax(y_hat, dim=1)
        target = target.long()
        loss = ce_loss(y_hat, target)
        print(loss)
        history.append(loss.detach().clone())
        loss.backward(retain_graph=True)
        optimizer.step()
    print(history[-1])
plt.plot(history)
plt.show()
