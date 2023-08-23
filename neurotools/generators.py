import math
import random
from typing import List

import numpy as np
import torch


class CategoricalDistributor:
    """
    helper class that distributes a given number of classes over the surface of a unit hypersphere. used to convert
    continuous action into logits.
    """

    def __init__(self, class_labels: List[int], device='cpu'):
        self.num_classes = 0
        self.labels = torch.Tensor(class_labels)
        self.loci = [torch.Tensor([math.cos(2 * math.pi / i), math.sin(2 * math.pi / i)])
                     for i in range(self.labels.shape[0])]

    def logits(self, selected):
        """
        :param selected: batch x 2 tensor of x, y coordinates
        :return: batch x num_classes distance from each locus along surface.
        """
        loci = torch.stack(self.loci)
        dist = torch.cdist(selected, loci, p=2.0)
        dist = dist.reshape((selected.shape[0], loci.shape[0]))
        return dist


class BasicMultiClassRL:
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
        self.device = dev
        self.ce = torch.nn.CrossEntropyLoss()
        self.min_loss = 0
        gt_ex = torch.zeros((label_max - label_min,))
        gt_ex[0] = 1
        bad_hat = torch.ones((label_max - label_min,))
        bad_hat[0] = 0
        self.max_loss = self.ce(bad_hat, gt_ex)
        self.chance_loss = self.ce(torch.ones((label_max - label_min)), gt_ex)
        self.selector = CategoricalDistributor(list(range(label_min, label_max)))
        self.class_templates = torch.Tensor([[[0, 1, 0, 0],
                                              [1, 1, 1, 1],
                                              [0, 1, 0, 0],
                                              [0, 1, 0, 0]],
                                             [[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]],
                                             [[0, 1, 0, 1],
                                              [1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [1, 0, 1, 0]],
                                             [[1, 1, 1, 1],
                                              [0, 0, 0, 0],
                                              [1, 1, 1, 1],
                                              [0, 0, 0, 0]],
                                             [[0, 1, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 1, 0, 0]]])
        self.prev_target = None

    def __len__(self):
        return self.num_samples

    def pop(self, batch_size: int = 1):
        # Generate a random label batch
        labels = torch.randint(self.label_min, self.label_max, (batch_size,))
        images = self.class_templates[labels].detach().clone()
        images = images + torch.normal(size=images.shape, mean=0, std=self.noise, device=self.device)
        return images, labels.long().to(self.device)

    def poll(self, batch=1, action=None):
        loss = None
        state, label = self.pop(batch)
        if action is not None:
            logits = self.selector.logits(action)
            loss = self.ce(logits, label)
        self.prev_target = label
        return state, loss

    def play(self, model, batch_size=1):
        jitter_samples = self.num_samples + random.randint(0, int(self.num_samples / 5))
        action = None
        for i in range(jitter_samples):
            state, loss = self.poll(action=action, batch=batch_size)
            if loss is not None:
                loss = loss / jitter_samples
            action = model.delta(state)
            yield state, loss
