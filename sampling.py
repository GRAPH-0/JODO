import numpy as np
import torch
from torch.nn import functional as F
import random
from models.utils import sample_combined_position_feature_noise, assert_mean_zero_with_mask, \
    sample_symmetric_edge_feature_noise, sample_gaussian_with_mask
from utils import *
from cond_gen import get_adj_matrix_fn
from mix_dpm_solver import DPM_Solver_hybrid


def mol_process(one_hot, x, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols"""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:n_nodes[i]]
        pos = pos[0:n_nodes[i]]
        if edge_types is not None:
            edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
            if formal_charges.shape[-1] != 0:
                fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
            else:
                fc = formal_charges[i][:n_nodes[i]].cpu().detach()
            mol_list.append((pos, atom_type, edge_type, fc))
        else:
            mol_list.append((pos, atom_type))

    return mol_list


def mol_process_2D(one_hot, formal_charges, n_nodes, edge_types=None):
    """Convert tensor to mols, without 3D position."""
    mol_list = []
    bs = one_hot.shape[0]
    for i in range(bs):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        atom_type = atom_type[0:n_nodes[i]]

        edge_type = edge_types[i][:n_nodes[i], :n_nodes[i]].cpu().detach()
        if formal_charges.shape[-1] != 0:
            fc = formal_charges[i][:n_nodes[i], 0].long().cpu().detach()
        else:
            fc = formal_charges[i][:n_nodes[i]].cpu().detach()
        mol_list.append((None, atom_type, edge_type, fc))

    return mol_list


def post_process(xh, atom_types, include_charge, node_mask, inverse_scaler,
                 edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    pos = xh[:, :, :3]
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, 3:-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, 3:]

    assert h_cat.shape[-1] == atom_types

    if edge_x is not None:
        pos, h_cat, h_int, h_edge = inverse_scaler(pos, h_cat, h_int, node_mask, edge_x, edge_mask)
    else:
        pos, h_cat, h_int = inverse_scaler(pos, h_cat, h_int, node_mask)
    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask
    if edge_x is not None:
        if compress_edge:
            edge_exist = h_edge[:, :, :, 0]
            edge_exist[edge_exist < 0.5] = 0.
            edge_exist[edge_exist >= 0.5] = 1.0
            edge_type = h_edge[:, :, :, 1] * 3.
            edge_type[edge_type >= 2.5] = 3.
            edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
            edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
            edge_type[edge_type < 0.5] = 0.
            edge_type = edge_exist * edge_type
            if h_edge.size(-1) == 3:
                edge_aromatic = h_edge[:, :, :, 2]
                edge_aromatic[edge_aromatic < 0.5] = 0.
                edge_aromatic[edge_aromatic >= 0.5] = 1.
                edge_aromatic = edge_exist * edge_aromatic
                edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
            h_edge = edge_type
        else:
            # all 0 set non-exist, others set argmax
            h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
            h_edge = torch.argmax(h_edge, dim=-1) + 1.0
            h_edge = h_edge_exist * h_edge

        return pos, h_cat, h_int, h_edge
    return pos, h_cat, h_int


def post_process_2D(xh, atom_types, include_charge, node_mask, inverse_scaler,
                    edge_x=None, edge_mask=None, compress_edge=False):
    """Split the xh [bs, n_nodes, pos_dim+atom_types+fc_charge], unormalize data"""
    if include_charge:
        h_int = xh[:, :, -1:]
        h_cat = xh[:, :, :-1]
    else:
        h_int = torch.zeros(0).to(xh.device)
        h_cat = xh[:, :, :]

    assert h_cat.shape[-1] == atom_types
    assert edge_x is not None

    _, h_cat, h_int, h_edge = inverse_scaler(None, h_cat, h_int, node_mask, edge_x, edge_mask)

    h_cat = F.one_hot(torch.argmax(h_cat, dim=2), atom_types) * node_mask
    h_int = torch.round(h_int).long() * node_mask

    if compress_edge:
        edge_exist = h_edge[:, :, :, 0]
        edge_exist[edge_exist < 0.5] = 0.
        edge_exist[edge_exist >= 0.5] = 1.0
        edge_type = h_edge[:, :, :, 1] * 3.
        edge_type[edge_type >= 2.5] = 3.
        edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.
        edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.
        edge_type[edge_type < 0.5] = 0.
        edge_type = edge_exist * edge_type
        if h_edge.size(-1) == 3:
            edge_aromatic = h_edge[:, :, :, 2]
            edge_aromatic[edge_aromatic < 0.5] = 0.
            edge_aromatic[edge_aromatic >= 0.5] = 1.
            edge_aromatic = edge_exist * edge_aromatic
            edge_type[torch.bitwise_and(edge_aromatic > 0., edge_type == 0.)] = 4.
        h_edge = edge_type
    else:
        # all 0 set non-exist, others set argmax
        h_edge_exist = torch.sum(h_edge > 0.5, dim=-1) != 0
        h_edge = torch.argmax(h_edge, dim=-1) + 1.0
        h_edge = h_edge_exist * h_edge

    return h_cat, h_int, h_edge


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def get_sampling_fn(config, noise_scheduler, nodes_dist, batch_size, n_samples, inverse_scaler,
                    eps=1e-3, prop_dist=None):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    only_2D = config.only_2D

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        if only_2D:
            sampler = AncestralSampler_2D(noise_scheduler, time_steps, config.model.pred_data, self_cond)
        else:
            sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, self_cond,
                                       get_self_cond_fn(config))
    elif config.sampling.method == 'fast':
        sampler = DPM_Solver_hybrid(noise_scheduler, config)
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model):
        model.eval()
        processed_mols = []
        with torch.no_grad():
            # random sample node first, then sort the number of nodes
            n_nodes_all = nodes_dist.sample(num_sampling_rounds * batch_size)
            # n_nodes_all = sorted(n_nodes_all, reverse=True)

            for r in range(num_sampling_rounds):
                # sample number of nodes
                # n_nodes = nodes_dist.sample(batch_size)
                n_nodes = n_nodes_all[r * batch_size:(r + 1) * batch_size]
                max_n_nodes = max(n_nodes)

                if prop_dist is not None:
                    context = prop_dist.sample_batch(n_nodes).to(device)
                else:
                    context = None

                # construct node and edge mask
                node_mask = torch.zeros(batch_size, max_n_nodes)
                for i in range(batch_size):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_combined_position_feature_noise(batch_size, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                # sample initial edge noise
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(batch_size, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                    # postprocessing
                    pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)
                else:
                    # sampling procedure
                    x_node = sampler.sampling(model, z, node_mask, edge_mask)
                    # postprocessing: split features and discretize and checking, and inverse
                    pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                assert_mean_zero_with_mask(pos, node_mask)

                # process tensors
                if pred_edge:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                else:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # shuffle mols and pick n_samples
        random.shuffle(processed_mols)
        return processed_mols[:n_samples]

    def sampling_fn_2D(model):
        model.eval()
        processed_mols = []
        with torch.no_grad():
            # random sample node first, then sort the number of nodes
            n_nodes_all = nodes_dist.sample(num_sampling_rounds * batch_size)
            # n_nodes_all = sorted(n_nodes_all, reverse=True)

            for r in range(num_sampling_rounds):
                # sample number of nodes
                # n_nodes = nodes_dist.sample(batch_size)
                n_nodes = n_nodes_all[r * batch_size:(r + 1) * batch_size]
                max_n_nodes = max(n_nodes)
                context = None

                # construct node and edge mask
                node_mask = torch.zeros(batch_size, max_n_nodes)
                for i in range(batch_size):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_gaussian_with_mask((batch_size, max_n_nodes, node_nf), device, node_mask)

                # sample initial edge noise
                edge_z = sample_symmetric_edge_feature_noise(batch_size, max_n_nodes, edge_nf, edge_mask)
                # sampling procedure
                x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                # postprocessing
                one_hot, fc, edge_types = post_process_2D(x_node, atom_types, include_fc, node_mask,
                                                               inverse_scaler, x_edge, edge_mask, compress_edge)

                # process tensors
                processed_mols += mol_process_2D(one_hot, fc, n_nodes, edge_types)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # shuffle mols and pick n_samples
        random.shuffle(processed_mols)
        return processed_mols[:n_samples]

    if only_2D:
        return sampling_fn_2D
    return sampling_fn


def get_cond_sampling_eval_fn(config, noise_scheduler, nodes_dist, batch_size, n_samples, inverse_scaler,
                              eps=1e-3, prop_dist=None, prop_norm=None):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    get_adj_matrix = get_adj_matrix_fn()
    mean, mad = prop_norm[config.cond_property]['mean'], prop_norm[config.cond_property]['mad']
    cond_property = config.cond_property
    outputNorm = {'mu': 1., 'alpha': 1, 'homo': 1000., 'lumo': 1000., 'gap': 1000, 'Cv': 1.}

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, self_cond,
                                   get_self_cond_fn(config))
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model, classifier):
        model.eval()
        classifier.eval()
        processed_mols = []
        loss_l1 = torch.nn.L1Loss(reduction='none')
        MAE_losses = []

        with torch.no_grad():
            # random sample node first, then sort the number of nodes
            n_nodes_all = nodes_dist.sample(num_sampling_rounds * batch_size)
            # n_nodes_all = sorted(n_nodes_all, reverse=True)

            for r in range(num_sampling_rounds):
                # sample number of nodes
                # n_nodes = nodes_dist.sample(batch_size)
                n_nodes = n_nodes_all[r * batch_size:(r + 1) * batch_size]
                max_n_nodes = max(n_nodes)

                if prop_dist is not None:
                    context = prop_dist.sample_batch(n_nodes).to(device)
                else:
                    context = None

                # construct node and edge mask
                node_mask = torch.zeros(batch_size, max_n_nodes)
                for i in range(batch_size):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_combined_position_feature_noise(batch_size, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                # sample initial edge noise
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(batch_size, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                    # postprocessing
                    pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)
                else:
                    # sampling procedure
                    x_node = sampler.sampling(model, z, node_mask, edge_mask)
                    # postprocessing: split features and discretize and checking, and inverse
                    pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                assert_mean_zero_with_mask(pos, node_mask)

                # process tensors
                # use the tensor as the input classifier
                bs, b_node, _ = pos.size()
                full_edges = get_adj_matrix(b_node, batch_size, device)

                pred = classifier(h0=one_hot.reshape(bs * b_node, -1), x=pos.reshape(bs * b_node, -1), edges=full_edges,
                                  edge_attr=None, node_mask=node_mask.reshape(bs * b_node, -1),
                                  edge_mask=edge_mask, n_nodes=b_node)

                # rescale the target
                assert context.size(-1) == 1
                target = context.clone().squeeze(-1)
                target = target * mad + mean
                pred = pred * mad + mean

                # calculate the l1 loss of output
                MAE_losses.append(loss_l1(pred, target))

                if pred_edge:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                else:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # shuffle mols and pick n_samples
        # random.shuffle(processed_mols)

        MAE_losses = torch.cat(MAE_losses)[:n_samples]
        mean_loss = MAE_losses.mean().item()
        return processed_mols[:n_samples], mean_loss * outputNorm[cond_property]

    return sampling_fn


def get_cond_multi_sampling_eval_fn(config, noise_scheduler, nodes_dist, batch_size, n_samples, inverse_scaler,
                                    eps=1e-3, prop_dist=None, prop_norm=None):
    device = config.device
    sampling_steps = config.sampling.steps
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond
    get_adj_matrix = get_adj_matrix_fn()
    cond_prop1 = config.cond_property1
    cond_prop2 = config.cond_property2
    mean1, mad1 = prop_norm[cond_prop1]['mean'], prop_norm[cond_prop1]['mad']
    mean2, mad2 = prop_norm[cond_prop2]['mean'], prop_norm[cond_prop2]['mad']
    outputNorm = {'mu': 1., 'alpha': 1, 'homo': 1000., 'lumo': 1000., 'gap': 1000, 'Cv': 1.}

    num_sampling_rounds = int(np.ceil(n_samples / batch_size))
    if config.sampling.method == 'ancestral':
        time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
        sampler = AncestralSampler(noise_scheduler, time_steps, config.model.pred_data, pred_edge, self_cond,
                                   get_self_cond_fn(config))
    else:
        raise ValueError('Invalid sampling method!')

    def sampling_fn(model, classifier1, classifier2):
        model.eval()
        classifier1.eval()
        classifier2.eval()
        processed_mols = []
        loss_l1 = torch.nn.L1Loss(reduction='none')
        MAE1_losses = []
        MAE2_losses = []

        with torch.no_grad():
            # random sample node first, then sort the number of nodes
            n_nodes_all = nodes_dist.sample(num_sampling_rounds * batch_size)
            # n_nodes_all = sorted(n_nodes_all, reverse=True)

            for r in range(num_sampling_rounds):
                # sample number of nodes
                # n_nodes = nodes_dist.sample(batch_size)
                n_nodes = n_nodes_all[r * batch_size:(r + 1) * batch_size]
                max_n_nodes = max(n_nodes)

                if prop_dist is not None:
                    context = prop_dist.sample_batch(n_nodes).to(device)
                else:
                    context = None

                # construct node and edge mask
                node_mask = torch.zeros(batch_size, max_n_nodes)
                for i in range(batch_size):
                    node_mask[i, 0:n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                # sample initial noise
                z = sample_combined_position_feature_noise(batch_size, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)

                # sample initial edge noise
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(batch_size, max_n_nodes, edge_nf, edge_mask)
                    # sampling procedure
                    x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context)
                    # postprocessing
                    pos, one_hot, fc, edge_types = post_process(x_node, atom_types, include_fc, node_mask,
                                                                inverse_scaler, x_edge, edge_mask, compress_edge)
                else:
                    # sampling procedure
                    x_node = sampler.sampling(model, z, node_mask, edge_mask)
                    # postprocessing: split features and discretize and checking, and inverse
                    pos, one_hot, fc = post_process(x_node, atom_types, include_fc, node_mask, inverse_scaler)

                assert_mean_zero_with_mask(pos, node_mask)

                # process tensors
                # use the tensor as the input classifier
                bs, b_node, _ = pos.size()
                full_edges = get_adj_matrix(b_node, batch_size, device)

                pred1 = classifier1(h0=one_hot.reshape(bs * b_node, -1), x=pos.reshape(bs * b_node, -1), edges=full_edges,
                                    edge_attr=None, node_mask=node_mask.reshape(bs * b_node, -1),
                                    edge_mask=edge_mask, n_nodes=b_node)
                pred2 = classifier2(h0=one_hot.reshape(bs * b_node, -1), x=pos.reshape(bs * b_node, -1), edges=full_edges,
                                    edge_attr=None, node_mask=node_mask.reshape(bs * b_node, -1),
                                    edge_mask=edge_mask, n_nodes=b_node)

                # rescale the target
                # context shape [B, 2]
                target1 = context[:, :1].clone().squeeze(-1)
                target1 = target1 * mad1 + mean1
                pred1 = pred1 * mad1 + mean1

                target2 = context[:, 1:].clone().squeeze(-1)
                target2 = target2 * mad2 + mean2
                pred2 = pred2 * mad2 + mean2

                # calculate the l1 loss of output
                MAE1_losses.append(loss_l1(pred1, target1))
                MAE2_losses.append(loss_l1(pred2, target2))

                if pred_edge:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes, edge_types)
                else:
                    processed_mols += mol_process(one_hot, pos, fc, n_nodes)
                print('Generate {}, Total {}.'.format(len(processed_mols), n_samples))

        # shuffle mols and pick n_samples
        # random.shuffle(processed_mols)

        MAE1_losses = torch.cat(MAE1_losses)[:n_samples]
        MAE2_losses = torch.cat(MAE2_losses)[:n_samples]
        mean1_loss = MAE1_losses.mean().item()
        mean2_loss = MAE2_losses.mean().item()
        return processed_mols[:n_samples], mean1_loss * outputNorm[cond_prop1], mean2_loss * outputNorm[cond_prop2]

    return sampling_fn

class AncestralSampler:
    """Ancestral sampling for 2D & 3D joint generation."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, pred_edge=False, self_cond=False,
                 cond_process_fn=None):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.pred_edge = pred_edge
        self.self_cond = self_cond
        self.cond_process_fn = cond_process_fn

    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.pred_edge:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                    cond_x, cond_edge_x = self.cond_process_fn(pred_t, edge_pred_t)
                else:
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                                context=context)

            else:
                if self.self_cond:
                    assert self.model_pred_data
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                   cond_x=cond_x, context=context)
                else:
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level, context=context)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_combined_position_feature_noise(bs, x_mean.shape[1], x_mean.shape[2] - 3, node_mask)

            # edge update
            if self.pred_edge:
                if self.model_pred_data:
                    edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                                  * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                         edge_pred_t.dim()) * edge_pred_t
                else:
                    edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t
                edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                         sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)

        assert_mean_zero_with_mask(x_mean[:, :, :3], node_mask)

        if self.pred_edge:
            return x_mean, edge_x_mean
        else:
            return x_mean


class AncestralSampler_2D:
    """Ancestral Sampler without 3D positions."""
    def __init__(self, noise_scheduler, time_steps, model_pred_data, self_cond=False):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = self_cond

    def sampling(self, model, z_T, node_mask, edge_mask, edge_z_T=None, context=None):
        x = z_T
        edge_x = edge_z_T
        bs = z_T.shape[0]
        cond_x, cond_edge_x = None, None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            # tmp = (1 - alpha_t_given_s**2) * c
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=x.device) * t

            noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            if self.self_cond:
                assert self.model_pred_data
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                cond_x, cond_edge_x = pred_t, edge_pred_t
            else:
                pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x, noise_level=noise_level,
                                            context=context)

            # node update
            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                sample_gaussian_with_mask(x.size(), x.device, node_mask)

            # edge update
            if self.model_pred_data:
                edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                              * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                     edge_pred_t.dim()) * edge_pred_t
            else:
                edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t

            edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                     sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)

        return x_mean, edge_x_mean
