import torch
import numpy
import networkx as nx
from neurotools import models, geometry, modules, util


class FuzzyMental:

    def __init__(self, target_beta_matrix, feature_names, atlas, roi_names, feature_generator, spatial, input_roi,
                 stim_frames):
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
        self.corr, self.idx_roi_map, self.rdms = geometry.pairwise_rsa(target_beta_matrix, atlas, min_roi_dim=5)
        self.ror_idx_map = {self.roi_names[idx]: i for i, idx in enumerate(self.idx_roi_map)}
        self.input_node = self.ror_idx_map[self.roi_names[input_roi]]
        self.stim_frames = stim_frames
        structure = nx.complete_graph(n=len(self.ror_idx_map), create_using=nx.DiGraph)
        for node in structure.nodes():
            structure.add_edge(node, node)
        self.reverb_model = models.ReverbNetwork(structure=structure,
                                                 node_shape=(1, 3, spatial, spatial),
                                                 edge_module=modules.WeightedConvolution,
                                                 inject_noise=True,
                                                 input_node=self.input_node,
                                                 track_activation_history=True)

    def beta_correlation_loss(self, run_list, verbose=True):
        """

        :param activation_states: Dictionary keyed on nodes with list of state tensors of same length as run list. All
                                  node keys must exist in `self.structure`.
        :param run_list: order in which conditions were presented to network (num_frames x 1)
        :param num_conditions:
        :param paradigm_index: index of this paradigm in graph rdm list attribute
        :return: loss scalar, the total dissimilarity between representations at each node in `self.brain` with
                 representations at corresponding nodes in `self.structure`.
        """
        # this design matrix holds for all parallel stimuli in batch this epoch
        design_matrix = torch.nn.functional.one_hot(run_list, num_classes=len(self.feature_names))
        loss = torch.Tensor([0.])

        # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
        for node, data in self.reverb_model.architecture.nodes(data=True):
            if node == -1:
                continue
            time_course = torch.stack(data["_past_states"], dim=1).view(1, len(run_list), -1)  # batch x t x n
            betas = torch.transpose(torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course,
                                    1,
                                    2)  # batch x n x k
            betas = torch.mean(betas, dim=0)  # average out batch dimension
            rdm = geometry.dissimilarity(betas, metric='dot')
            target_brain_rdm = torch.Tensor(self.rdms[node])
            # compute the linear correlation between the model rdm at this node and the measured rdm at this roi
            local_loss = util.pearson_correlation(rdm, target_brain_rdm)
            loss = loss + local_loss
        # scale so loss stay comparable across prunings
        loss = loss / len(self.reverb_model.nodes)
        if verbose:
            print("computed beta coefficients")
        return loss

    def fit(self, local_epochs, meta_epochs, lr=0.01):
        optimizer = torch.optim.Adam(lr=lr, params=self.reverb_model.parameters())
        for meta in range(meta_epochs):
            epoch_history = []
            epoch = 0
            while not util.is_converged(epoch_history, abs_tol=.01, consider=20) and epoch < local_epochs:
                epoch += 1
                batch, runlist = self.stim_gen.get_batch()
                runlist = torch.Tensor(runlist)
                self.reverb_model.detach()
                optimizer.zero_grad()
                for stim in batch:
                    for i in range(self.stim_frames):
                        self.reverb_model.architecture.nodes[-1]['state'] = stim
                        self.reverb_model.forward()
                loss = self.beta_correlation_loss(runlist)
                epoch_history.append(loss.detach().cpu().item())
                loss.backward()
                optimizer.step()
