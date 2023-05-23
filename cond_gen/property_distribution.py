import torch
from torch.distributions.categorical import Categorical

class DistributionProperty:
    def __init__(self, dataset, prop2idx, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = list(prop2idx.keys())
        self.n_prop = len(self.properties)

        for prop in self.properties:
            self.distributions[prop] = {}

        # iterate dataset, get data nodes and corresponding properties
        num_atoms = []
        prop_values = []
        prop_ids = torch.tensor(list(prop2idx.values()))
        for idx in range(len(dataset.indices())):
            data = dataset.get(dataset.indices()[idx])
            tars = []
            for prop_id in prop_ids:
                if prop_id == 11:
                    tars.append(dataset.sub_Cv_thermo(data).reshape(1))
                else:
                    tars.append(data.y[0][prop_id].reshape(1))
            tars = torch.cat(tars)
            num_atoms.append(data.num_atom)
            prop_values.append(tars)
        num_atoms = torch.cat(num_atoms)  # [N]
        prop_values = torch.stack(prop_values)  # [N, num_prop]

        self._create_prob_dist(num_atoms, prop_values)
        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]

            if values_filtered.size(0) > 0:
                self._create_prob_given_nodes(values_filtered, n_nodes)
                # self.distributions[][n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values, n_nodes):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values, dim=0).values, torch.max(values, dim=0).values
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros((self.n_prop, n_bins))
        for i in range(values.size(0)):
            val = values[i]
            idx = ((val - prop_min)/prop_range * n_bins).long()
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            idx[idx==n_bins] = n_bins - 1
            idx = idx.unsqueeze(-1)
            histogram.scatter_add_(1, idx, torch.ones_like(idx, dtype=histogram.dtype))
        # iterate all props
        probs = histogram / torch.sum(histogram, dim=-1, keepdim=True)
        for i, prop in enumerate(self.properties):
            self.distributions[prop][n_nodes] = {
                'probs': Categorical(probs[i]),
                'params': [prop_min[i], prop_max[i]]
            }

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val
