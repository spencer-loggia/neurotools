import random

import torch


class CategoricalDistributor:
    """
    helper class that distributes a given number of classes over the surface of a unit hypersphere. used to convert
    continuous action into logits.
    """
    def __init__(self, initial_classes, device='cpu'):
        self.num_classes = 0
        self.loci = []
        self.names = []
        self.device = device
        for name in initial_classes:
            self.add_class(name=name)
        self.optim = torch.optim.Adam(lr=.001, params=self.loci)

    def add_class(self, name=None):
        self.num_classes += 1
        locus = torch.nn.Parameter((torch.rand(size=(2,)) * 2) * torch.pi) # range: 0, 2*pi
        self.loci.append(locus)
        self.names.append(name)

    def logits(self, polar):
        """
        :param polar: batch x 2 tensor of theta, psi spherical coordinates
        :return: batch x num_classes distance from each locus along surface.
        """
        loci = torch.stack(self.loci).to(self.device)
        polar = torch.remainder(polar, torch.pi * 2)
        theta = polar[:, 0][:, None]
        theta2 = loci[:, 0][None, :]
        psi = polar[:, 1][:, None]
        psi2 = loci[:, 1][None, :]
        arc = torch.acos(torch.sin(theta)*torch.sin(theta2) + torch.cos(theta) * torch.cos(theta2) * torch.cos(psi - psi2))
        arc[arc > torch.pi] = 2 * torch.pi - arc[arc > torch.pi]
        return arc


class BasicMultiClass:
    """
    five synthetic 4x4 classes for testing
    """

    def __init__(self, num_samples=100, noise=.2, label_min=0, label_max=5, dev='cpu'):
        self.res = 4
        self.class_names = ["cross", "diag", "checker", "horiz", "vert"]
        self.num_samples = num_samples
        self.noise = noise
        self.label_min = label_min
        self.label_max = label_max
        self.selector = CategoricalDistributor(initial_classes=self.class_names[label_min:label_max], device=dev)
        self.device = dev
        self.ce = torch.nn.CrossEntropyLoss()
        self.min_loss = 0
        gt_ex = torch.zeros((label_max - label_min,))
        gt_ex[0] = 1
        bad_hat = torch.ones((label_max - label_min,))
        bad_hat[0] = 0
        self.max_loss = self.ce(bad_hat, gt_ex)
        self.chance_loss = self.ce(torch.ones((label_max - label_min)), gt_ex)
        self.prev_target = None

    def __len__(self):
        return self.num_samples

    def pop(self):
        # Generate a random label
        label = torch.randint(self.label_min, self.label_max, (1,))

        if label == 0:
            # Generate an up/down cross image
            image = torch.tensor([[0, 1, 0, 0],
                                  [1, 1, 1, 1],
                                  [0, 1, 0, 0],
                                  [0, 1, 0, 0]], device=self.device)
        elif label == 1:
            # Generate a diagonal line image
            image = torch.tensor([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], device=self.device)
        elif label == 2:
            # Generate a checkerboard image
            image = torch.tensor([[0, 1, 0, 1],
                                  [1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [1, 0, 1, 0]], device=self.device)
        elif label == 3:
            # Generate horizontal lines image
            image = torch.tensor([[1, 1, 1, 1],
                                  [0, 0, 0, 0],
                                  [1, 1, 1, 1],
                                  [0, 0, 0, 0]], device=self.device)
        else:
            # Generate a vertical line image
            image = torch.tensor([[0, 1, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 1, 0, 0]], device=self.device)

        image = image + torch.normal(size=image.shape, mean=0, std=self.noise, device=self.device)

        return image, label.long().to(self.device)

    def poll(self, action=None):
        loss = None
        state, label = self.pop()
        if action is not None:
            logits = self.selector.logits(action)
            loss = self.ce(logits, self.prev_target)
        self.prev_target = label
        return state, loss

    def play(self, model):
        jitter_samples = self.num_samples + random.randint(0, int(self.num_samples / 5))
        action = None
        for i in range(jitter_samples):
            state, loss = self.poll(action)
            if loss is not None:
                loss = loss / jitter_samples
            action = model.delta(state)
            yield state, loss
