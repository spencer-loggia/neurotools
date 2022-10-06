import torch
import numpy
import networkx as nx
from neurotools import models, geometry, modules, util


def _set_mp_env():
    torch.multiprocessing.set_start_method('spawn')


class FuzzyMental:

    def __init__(self, target_beta_matrix, feature_names, atlas, roi_names, feature_generator, spatial, input_roi,
                 stim_frames, generations=20, population=5, max_iter=1000):
        """
        Designed to fit a Reverb Network to brain data.
        :param target_beta_matrix: (voxels x features) from processed mri data
        :param feature_names: (features) list
        :param atlas: (voxels) roi index
        :param stim_frames: number of time steps to allow network for each frame from stim_gen
        :param roi_names: (rois) roi names
        :param feature_generator: a class that must implement a "get_batch" method that yields
                                  tuple[(n, channel, spatial, spatial) Tensor, length n list of feature indexes] where
                                  the feature indexes correspond to those in feature_names and target_beta_matrix, and
                                  each feature in feature names is represented at least once.
        """
        assert len(feature_names) == target_beta_matrix.shape[-1]
        self.target_beta_matrix = target_beta_matrix
        self.feature_names = feature_names
        self.atlas = atlas
        self.roi_names = roi_names
        self.stim_gen = feature_generator
        self.corr, self.idx_roi_map, self.rdms = geometry.pairwise_rsa(target_beta_matrix, atlas,
                                                                       min_roi_dim=5, ignore_atlas_base=False)
        self.ror_idx_map = {self.roi_names[idx]: i for i, idx in enumerate(self.idx_roi_map)}
        self.input_node = self.ror_idx_map[self.roi_names[input_roi]]
        self.stim_frames = stim_frames
        self.generations = generations
        self.population = population
        self.max_iter = max_iter
        structure = nx.complete_graph(n=len(self.ror_idx_map), create_using=nx.DiGraph)
        for node in structure.nodes():
            structure.add_edge(node, node)
        self.reverb_model = models.ReverbNetwork(structure=structure,
                                                 node_shape=(1, 3, spatial, spatial),
                                                 edge_module=modules.WeightedConvolution,
                                                 inject_noise=True,
                                                 input_node=self.input_node,
                                                 track_activation_history=True)

    def beta_correlation_loss(self, model, run_list, verbose=True):
        """
        computes how well the network model matches functional representations in the brain
        :param model: the reverb model
        :param run_list: order in which conditions were presented to network (num_frames x 1)
        :return: loss scalar, the total dissimilarity between representations at each node in `self.brain` with
                 representations at corresponding nodes in `self.structure`.
        """
        # this design matrix holds for all parallel stimuli in batch this epoch
        design_matrix = torch.nn.functional.one_hot(run_list, num_classes=len(self.feature_names)).float()
        loss = torch.Tensor([0.])

        # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
        for node, data in model.architecture.nodes(data=True):
            if node == -1:
                continue
            time_course = torch.stack(data["_past_states"], dim=1).view(1, len(run_list), -1)  # batch x t x n
            betas = torch.transpose(torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course,
                                    1,
                                    2)  # batch x n x k
            betas = torch.mean(betas, dim=0).unsqueeze(0)  # average out batch dimension
            rdm = geometry.dissimilarity(betas, metric='dot').squeeze()
            target_brain_rdm = torch.Tensor(self.rdms[node]).squeeze()
            # compute the linear correlation between the model rdm at this node and the measured rdm at this roi
            local_loss = util.pearson_correlation(rdm, target_brain_rdm)
            loss = loss + local_loss
        # scale so loss stay comparable across prunings
        loss = loss / len(model.architecture.nodes)
        if verbose:
            print("computed beta coefficients")
        return loss

    def _fit(self, reverb_model, lr=0.001, verbose=True):
        optimizer = torch.optim.Adam(lr=lr, params=reverb_model.parameters())
        epoch_history = []
        epoch = 0
        while not util.is_converged(epoch_history, abs_tol=.001, consider=100) and epoch < self.max_iter:
            epoch += 1
            batch, cond_list = self.stim_gen.get_batch()
            reverb_model.detach()
            optimizer.zero_grad()
            runlist = []
            for idx, stim in enumerate(batch):
                for i in range(self.stim_frames):
                    runlist.append(cond_list[idx])
                    reverb_model.architecture.nodes[-1]['state'] = stim.unsqueeze(0)
                    reverb_model.forward()
            loss = self.beta_correlation_loss(reverb_model, torch.Tensor(runlist).long())
            epoch_history.append(loss.detach().cpu().item())
            if verbose:
                print("SL: Loss epoch", epoch, "is", epoch_history[-1])
            # we want to maximize correlation so we negate loss
            (-1 * loss).backward()
            optimizer.step()
        return epoch_history[-1]

    def fit(self, mp=True):
        _set_mp_env()
        for generation in range(self.generations):
            print("**** SS: Generation", generation, "of", self.generations, "****")
            pop = [self.reverb_model.clone().mutate() for _ in range(self.population)]
            if mp:
                with torch.multiprocessing.Pool() as p:
                    res = p.map(self._fit, pop)
            else:
                res = []
                for m in pop:
                    res.append(self._fit(m))
            res = torch.Tensor(res)
            print("SS: Generation fitness: ", res.detach().cpu().tolist())
            # choose model with highest correlation
            best_idx = torch.argmax(torch.nan_to_num(res, nan=-1, posinf=-1, neginf=-1))
            self.reverb_model = pop[best_idx]

