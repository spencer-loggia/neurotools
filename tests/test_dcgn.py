from neurotools import environments
import torch
from stim_gen import GratingGenerator


def test_elegant_dcgn():
    n_conditions = 3
    roi_1 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([1, 2, 0]).unsqueeze(0)
    roi_2 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([2, 1, 0]).unsqueeze(0)
    roi_3 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([0, 0, 1]).unsqueeze(0)
    roi_4 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([0, -1, 1]).unsqueeze(0)
    atlas = torch.Tensor(([1] * 10) + ([2] * 10) + ([3] * 10) + ([4] * 10))
    betas = torch.cat([roi_1, roi_2, roi_3, roi_4], dim=0)
    stim_gen = GratingGenerator(((2., (0., 1., 0.), (0., -1., 0.)),
                                 (2., (1., 0., 0.), (0., 0., -1.)),
                                 (4., (0., 1., 0.), (0., -1., 0.))), res=32)
    dcgn = environments.ElegantFuzzyMental(betas, ["condition1", "condition2", "condition3"], atlas=atlas,
                                           roi_names={1: "node1", 2: "node2", 3: "node3", 4: "node4"},
                                           feature_generator=stim_gen, spatial=32,
                                           stim_frames=6, max_iter=2000, device='cuda', sparsity=1.0)
    dcgn.fit()
    print(dcgn.reverb_model.edge.out_edge.squeeze().detach())


def test_dcgn():
    n_conditions = 3
    roi_1 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([1, 2, 0]).unsqueeze(0)
    roi_2 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([2, 1, 0]).unsqueeze(0)
    roi_3 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([0, 0, 1]).unsqueeze(0)
    roi_4 = torch.normal(size=(10, n_conditions), mean=0, std=1) + torch.Tensor([0, -1, 1]).unsqueeze(0)
    atlas = torch.Tensor(([1] * 10) + ([2] * 10) + ([3] * 10) + ([4] * 10))
    betas = torch.cat([roi_1, roi_2, roi_3, roi_4], dim=0)
    stim_gen = GratingGenerator(((2., (0., 1., 0.), (0., -1., 0.)),
                                 (2., (1., 0., 0.), (0., 0., -1.)),
                                 (4., (0., 1., 0.), (0., -1., 0.))), res=32)
    dcgn = environments.FuzzyMental(betas, ["condition1", "condition2", "condition3"], atlas=atlas,
                                    roi_names={1: "node1", 2: "node2", 3: "node3", 4: "node4"},
                                    feature_generator=stim_gen, spatial=32,
                                    input_roi=1, stim_frames=6, generations=1, max_iter=1000, population=1,
                                    device='cuda')
    dcgn.fit(mp=False)


if __name__ == '__main__':
    test_elegant_dcgn()
