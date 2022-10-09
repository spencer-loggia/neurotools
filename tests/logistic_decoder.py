from neurotools import models, modules
import torch
import torchvision
from torchvision.datasets.mnist import FashionMNIST
import networkx as nx
import numpy as np
import pickle

batch_size_train = 200

train_loader = torch.utils.data.DataLoader(
    FashionMNIST('../tmp/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
dev = 'cuda'
ce_loss = torch.nn.NLLLoss()
decoder = torch.nn.Conv2d(kernel_size=28, in_channels=1, out_channels=10).to(dev)
optimizer = torch.optim.Adam(lr=.001, params=decoder.parameters())
for epoch in range(10):
    print("\n******************\nEPOCH", epoch, "\n*********************\n")
    for i, (stims, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        stims = stims.to(dev)
        targets = targets.to(dev)
        y_hat = decoder(stims)
        y_hat = torch.log_softmax(y_hat, dim=1)
        pred = torch.argmax(y_hat, dim=1).view(batch_size_train, 1, 1)
        targets = targets.long().view(batch_size_train, 1, 1)
        correct = (pred == targets).detach().cpu()
        print(torch.sum(correct) / batch_size_train)
        local_loss = ce_loss(y_hat, targets.to(dev))
        local_loss.backward()
        optimizer.step()

with open("../models/linear_decoder_fashion_mnist.pkl", 'wb') as f:
    pickle.dump(decoder, f)

