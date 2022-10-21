from neurotools import models
from neurotools import modules
import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import networkx as nx
from matplotlib import pyplot as plt

batch_size_train = 1
batch_size_test = 1

revnet = models.ElegantReverbNetwork(num_nodes=10, input_nodes=(0,), node_shape=(1, 2, 28, 28), edge_module=modules.ElegantReverb, device='cuda')

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
                                     torch.nn.Conv2d(kernel_size=14, in_channels=1, out_channels=10, device="cuda"))
present_frames = 10
optimizer = torch.optim.Adam(lr=.01, params=list(revnet_decoder.parameters()) + list(revnet.parameters()))
ce_loss = torch.nn.NLLLoss()
history = []
for epoch in range(20):
    for i, (stim, target) in enumerate(train_loader):
        optimizer.zero_grad()
        revnet.detach(reset_intrinsic=True)
        if i > 500:
            break
        for i in range(present_frames):
            revnet(stim.to("cuda"))
        decode_input = revnet.states[0, 0, :, :][None, None, :, :].clone()
        y_hat = revnet_decoder(decode_input)
        y_hat = y_hat.view(1, 10)
        y_hat = torch.log_softmax(y_hat, dim=1)
        target = target.long().to("cuda")
        loss = ce_loss(y_hat[-100:], target[-100:])
        print(loss)
        history.append(loss.detach().cpu().item())
        loss.backward(retain_graph=True)
        optimizer.step()
    print(history[-1])
plt.plot(history)
plt.show()
