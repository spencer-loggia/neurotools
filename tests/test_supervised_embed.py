from neurotools.models import SupervisedEmbed
import torch
import numpy as np
from matplotlib import pyplot as plt


def test_supervised_embed():
    data = torch.normal(mean=0, std=1, size=(100, 10))
    data = (data - data.mean(dim=1).unsqueeze(1)) / data.std(dim=1).unsqueeze(1)
    targets = torch.from_numpy(np.random.randint(0, 2, size=(100,)))
    super_embed = SupervisedEmbed(n_components=2, device='cpu')
    super_embed.fit(data, targets, max_iter=10000, verbose=True)
    embed = super_embed.predict(data)
    components = super_embed.get_components()
    plt.scatter(embed[:, 0], embed[:, 1])
    print(components)
    plt.show()

if __name__=='__main__':
    test_supervised_embed()