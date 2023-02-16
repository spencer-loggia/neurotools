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

    def add_class(self, name=None):
        self.num_classes += 1
        locus = (torch.rand(size=(2,)) * 2) * torch.pi # range: 0, 2*pi
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

    def __init__(self, num_samples=1000, noise=.2, label_min=0, label_max=5, dev='cpu'):
        self.res = 4
        self.class_names = ["cross", "diag", "checker", "horiz", "vert"]
        self.num_samples = num_samples
        self.noise = noise
        self.label_min = label_min
        self.label_max = label_max
        self.selector = CategoricalDistributor(initial_classes=self.class_names[label_min:label_max])
        self.prev_class = None
        self.device = dev
        self.ce = torch.nn.NLLLoss()

    def __len__(self):
        return self.num_samples

    def pop(self):
        # Generate a random label
        label = torch.randint(self.label_min, self.label_max, (1,)).item()

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
                                  [0, 1, 0 , 1],
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
        state, label = self.pop()
        if action is None:
            return state, None
        elif self.prev_class is None:
            raise RuntimeError("No previous state was presented, yet an action was provided.")
        else:
            logits = self.selector.logits(action)
            probs = torch.log_softmax(logits, dim=1)
            loss = self.ce(probs, label.long())
        return state, loss
