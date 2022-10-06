from neurotools import environments
import torch
from stim_gen import GratingGenerator


def test_dcgn():
    n_conditions = 3
    roi_1 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([1, 2, 0]).unsqueeze(0)
    roi_2 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([2, 1, 0]).unsqueeze(0)
    roi_3 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([0, 0, 1]).unsqueeze(0)
    atlas = torch.Tensor(([0]*10) + ([1]*10) + ([2]*10))
    betas = torch.cat([roi_1, roi_2, roi_3], dim=0)
    stim_gen = GratingGenerator(((2., (0., 1., 0.), (0., -1., 0.)),
                                 (2., (1., 0., 0.), (0., 0., -1.)),
                                 (4., (0., 1., 0.), (0., -1., 0.))), res=32)
    dcgn = environments.FuzzyMental(betas, ["condition1", "condition2", "condition3"], atlas=atlas,
                                    roi_names=["node1", "node2", "node3"], feature_generator=stim_gen, spatial=32,
                                    input_roi=0, stim_frames=6, generations=20, max_iter=100, population=5)
    dcgn.fit(mp=True)


if __name__=='__main__':
    test_dcgn()