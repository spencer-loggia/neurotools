import torch
import sys
import networkx as nx
from neurotools import models, geometry, modules, util


def _set_mp_env():
    torch.multiprocessing.set_start_method('spawn')


class ElegantFuzzyMental:
    def __init__(self, target_beta_matrix, feature_names, atlas, roi_names: dict, feature_generator, spatial,
                 stim_frames, max_iter=1000, input_nodes=(1,), device='cpu', sparsity=0.1, beta_mode='last'):
        """
        Designed to fit an ElegantReverbNetwork to braind data.
        :param target_beta_matrix: (voxels x features) from processed mri data
        :param feature_names: (features) list
        :param atlas: (voxels) roi index. index of zero is taken to mean background and is not included.
        :param stim_frames: number of time steps to allow network for each frame from stim_gen
        :param roi_names: (rois) roi names
        :param feature_generator: a class that must implement a "get_batch" method that yields
                                  tuple[(n, channel, spatial, spatial) Tensor, length n list of feature indexes] where
                                  the feature indexes correspond to those in feature_names and target_beta_matrix, and
                                  each feature in feature names is represented at least once.m
        :param beta_mode: what data to use from time series when computing betas
        """
        assert len(feature_names) == target_beta_matrix.shape[-1]
        self.feature_names = feature_names
        self.atlas = atlas
        self.roi_names = roi_names
        self.device = device
        self.stim_gen = feature_generator
        beta_roi_list, self.idx_roi_map = util.atlas_to_list(target_beta_matrix, atlas, min_dim=5,
                                                             ignore_atlas_base=True)
        self.corr, self.rdms = geometry.pairwise_rsa(beta_roi_list, rdm_metric='cosine', pairwise_metric='spearman')

        loc_input_nodes = []

        self.ror_idx_map = {self.roi_names[idx.item()]: i for i, idx in enumerate(self.idx_roi_map)}
        self.stim_frames = stim_frames

        for input_node in input_nodes:
            loc_input_nodes.append(self.ror_idx_map[self.roi_names[input_node]])
        if len(loc_input_nodes) == 0:
            self.input_nodes = None
        else:
            self.input_nodes = input_nodes

        self.sparcity = sparsity
        self.beta_mode = beta_mode
        self.max_iter = max_iter
        self.num_nodes = len(self.ror_idx_map)
        self.reverb_model = models.ElegantReverbNetwork(num_nodes=self.num_nodes,
                                                        node_shape=(1, 3, spatial, spatial),
                                                        edge_module=modules.ElegantWeightedConvolution,
                                                        inject_noise=True,
                                                        input_nodes=self.input_nodes,
                                                        track_activation_history=True,
                                                        device=device)
        self.corr = self.corr.to(device)
        self.rdms = self.rdms.to(device)

    def beta_correlation(self, run_list, verbose=True, struct_w=.5, rdm_w=1.):
        # this design matrix holds for all parallel stimuli in batch this epoch
        design_matrix = torch.nn.functional.one_hot(run_list, num_classes=len(self.feature_names) + 1).float().to(
            self.device)
        design_matrix = design_matrix[:, 1:] # get rid of base column
        # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
        time_course = torch.stack(self.reverb_model.past_states, dim=0)[:, 1:].clone()  # ignore stim node
        time_course = time_course.view(len(design_matrix), self.num_nodes, -1).transpose(0, 1) # node x t x spatial
        dm_base = (torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T).unsqueeze(0)  # feature x t
        betas = torch.matmul(dm_base, time_course)
        betas = betas.transpose(1, 2)
        model_corr, model_rdms = geometry.pairwise_rsa(betas, rdm_metric='dot', pairwise_metric='pearson')
        rdm_corr_obj = util.pearson_correlation(model_rdms, self.rdms, dim=1)
        rdm_corr_obj = torch.mean(rdm_corr_obj)
        structural_obj = util.pearson_correlation(model_corr, self.corr)
        loss = rdm_w * rdm_corr_obj + struct_w * structural_obj
        if verbose:
            print("SL: computed model beta coefficients and pairwise rdms")
            print("mean RDM correlation:", rdm_corr_obj.detach().cpu().item())
            print("structural correlation:", structural_obj.detach().cpu().item())
        return loss

    def fit(self, lr=.01, verbose=True):
        optimizer = torch.optim.Adam(lr=lr, params=self.reverb_model.parameters())
        epoch_history = []
        epoch = 0
        while not util.is_converged(epoch_history, abs_tol=.0005, consider=1000) and epoch < self.max_iter:
            epoch += 1
            batch, cond_list = self.stim_gen.get_batch()
            self.reverb_model.detach()
            optimizer.zero_grad()
            runlist = []
            for idx, stim in enumerate(batch):
                stim = stim.to(self.device)
                for i in range(self.stim_frames):
                    if self.beta_mode == 'all':
                        runlist.append(cond_list[idx])
                    elif self.beta_mode == 'last':
                        runlist.append(-1)
                    self.reverb_model.forward(stim.unsqueeze(0))
                runlist[-1] = cond_list[idx]
            loss = self.beta_correlation(torch.Tensor(runlist).long() + 1, verbose=verbose)
            l1 = self.sparcity * self.reverb_model.l1()
            print("edge weight l1:", l1.detach().cpu().item())
            # the correlation portion of the loss must be negated
            loss = -1 * loss + l1
            epoch_history.append(loss.detach().cpu().item())
            if verbose:
                print("Epoch", epoch)
                # flush the buffer every 10 epoch so we get some update
                # this forces process comm. so slow things up if we do too often
                if (epoch % 10) == 0:
                    sys.stdout.flush()
            loss.backward()
            optimizer.step()
        return epoch_history


class FuzzyMental:

    def __init__(self, target_beta_matrix, feature_names, atlas, roi_names: dict, feature_generator, spatial, input_roi,
                 stim_frames, generations=20, population=5, max_iter=1000, device='cpu'):
        """
        Designed to fit a Reverb Network to brain data.
        :param target_beta_matrix: (voxels x features) from processed mri data
        :param feature_names: (features) list
        :param atlas: (voxels) roi index. index of zero is taken to mean background and is not included.
        :param stim_frames: number of time steps to allow network for each frame from stim_gen
        :param roi_names: (rois) roi names
        :param feature_generator: a class that must implement a "get_batch" method that yields
                                  tuple[(n, channel, spatial, spatial) Tensor, length n list of feature indexes] where
                                  the feature indexes correspond to those in feature_names and target_beta_matrix, and
                                  each feature in feature names is represented at least once.m
        """
        assert len(feature_names) == target_beta_matrix.shape[-1]
        self.target_beta_matrix = target_beta_matrix
        self.feature_names = feature_names
        self.atlas = atlas
        self.roi_names = roi_names
        self.device = device
        self.stim_gen = feature_generator
        beta_roi_list, self.idx_roi_map = util.atlas_to_list(target_beta_matrix, atlas, min_dim=5,
                                                             ignore_atlas_base=True)
        self.corr, self.rdms = geometry.pairwise_rsa(beta_roi_list, rdm_metric='cosine', pairwise_metric='spearman')
        self.ror_idx_map = {self.roi_names[idx.item()]: i for i, idx in enumerate(self.idx_roi_map)}
        if not isinstance(input_roi, tuple) and not isinstance(input_roi, list):
            input_roi = [input_roi]
        self.input_node = []
        for r in input_roi:
            self.input_node.append(self.ror_idx_map[self.roi_names[r]])
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
                                                 input_nodes=self.input_node,
                                                 track_activation_history=True,
                                                 device=device)
        self.corr = self.corr.to(device)
        self.rdms = self.rdms.to(device)

    def beta_correlation(self, model, run_list, verbose=True):
        """
        computes how well the network model matches functional representations in the brain
        :param model: the reverb model
        :param run_list: order in which conditions were presented to network (num_frames x 1)
        :return: loss scalar, the total dissimilarity between representations at each node in `self.brain` with
                 representations at corresponding nodes in `self.structure`.
        """
        # this design matrix holds for all parallel stimuli in batch this epoch
        design_matrix = torch.nn.functional.one_hot(run_list, num_classes=len(self.feature_names)).float().to(
            self.device)
        # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
        beta_list = []
        for node, data in model.architecture.nodes(data=True):
            if node == -1:
                continue
            time_course = torch.stack(data["_past_states"], dim=1).view(1, len(run_list), -1)  # batch x t x n
            betas = torch.transpose(
                        torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course,
                    1, 2)  # batch x n x k
            betas = torch.mean(betas, dim=0).unsqueeze(0)  # average out batch dimension
            beta_list.append(betas)
        model_corr, model_rdms = geometry.pairwise_rsa(beta_list, rdm_metric='dot', pairwise_metric='pearson')
        rdm_corr_obj = util.pearson_correlation(model_rdms, self.rdms, dim=1)
        rdm_corr_obj = torch.mean(rdm_corr_obj)
        structural_obj = util.pearson_correlation(model_corr, self.corr)
        loss = rdm_corr_obj + structural_obj
        if verbose:
            print("SL: computed model beta coefficients and pairwise rdms")
            print("mean RDM correlation:", rdm_corr_obj.detach().cpu().item())
            print("structural correlation:", structural_obj.detach().cpu().item())
        return loss

    def _fit(self, reverb_model, lr=0.01, verbose=True):
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
                stim = stim.to(self.device)
                for i in range(self.stim_frames):
                    runlist.append(cond_list[idx])
                    reverb_model.architecture.nodes[-1]['state'] = stim.unsqueeze(0)
                    reverb_model.forward()
            loss = self.beta_correlation(reverb_model, torch.Tensor(runlist).long(), verbose=verbose)
            epoch_history.append(loss.detach().cpu().item())
            if verbose:
                print("Epoch", epoch)
                # flush the buffer every 10 epoch so we get some update
                # this forces process comm. so slow things up if we do too often
                if (epoch % 10) == 0:
                    sys.stdout.flush()
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
            res = torch.Tensor(res) / 2
            print("SS: Generation fitness: ", res.detach().cpu().tolist())
            # choose model with highest correlation
            best_idx = torch.argmax(torch.nan_to_num(res, nan=-1, posinf=-1, neginf=-1))
            self.reverb_model = pop[best_idx]


class FuzzyRL:

    def __init__(self, feature_names, state_generator, spatial, num_nodes=4, stim_frames=1,
                 max_iter=1000, input_nodes=(1,), feedback_nodes=(2,), output_nodes=(3,), device='cpu'):
        self.feature_names = feature_names
        self.state_generator = state_generator
        self.spatial = spatial
        if self.spatial != state_generator.res:
            raise ValueError("state generator resolution must equal network spatial dimensionality.")
        self.cycles_per_stim = stim_frames
        self.input_nodes = input_nodes
        self.feedback_nodes = feedback_nodes
        self.output_nodes = output_nodes
        self.device = device
        self.max_iter = max_iter
        self.model = models.ElegantReverbNetwork(num_nodes=num_nodes, input_nodes=input_nodes, device=self.device)

    def fit(self):
        pass
