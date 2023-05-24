import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from random import random
from utils import *

from models.utils import remove_mean_with_mask, check_mask_correct, \
    assert_mean_zero_with_mask, assert_correctly_masked, sample_combined_position_feature_noise, \
    sample_symmetric_edge_feature_noise, sample_gaussian_with_mask


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def gradient_clipping(params, gradnorm_queue, max_grad, disable_log):
    if max_grad <= 1.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad)
        return

    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()
    max_grad_norm = min(max_grad_norm, max_grad)

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        params, max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if not disable_log:
        if float(grad_norm) > 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std():
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
    return grad_norm


class Queue():
    # from EDM
    # Gradient clipping
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def optimization_manager(config):
    """Return an optimize_fn based on `config`."""

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    disable_log = config.optim.disable_grad_log

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimize with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            # torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            gradient_clipping(params, gradnorm_queue, grad_clip, disable_log)
        optimizer.step()

    return optimize_fn


def get_step_fn(noise_scheduler, train, optimize_fn, scaler, config, prop_dist=None):
    if config.pred_edge:
        if config.only_2D:
            loss_fn = get_sde_2D_loss_fn(noise_scheduler, train, scaler, config)
        else:
            loss_fn = get_sde_graph_loss_fn(noise_scheduler, train, scaler, config, prop_dist)
    else:
         loss_fn = get_sde_node_loss_fn(noise_scheduler, train, scaler, config)

    def step_fn(state, batch):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
        return loss

    return step_fn


def get_sde_node_loss_fn(noise_scheduler, train, scaler, config):
    """The loss function for node features and positions."""
    device = config.device
    include_charges = config.model.include_fc_charge
    pred_data = config.model.pred_data
    reduce_mean = config.training.reduce_mean

    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    noise_align = config.model.noise_align
    self_cond = config.model.self_cond

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data
        xh, node_mask, edge_mask = process_batch(batch, device, include_charges, scaler)

        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)
        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        noise = sample_combined_position_feature_noise(xh.shape[0], xh.shape[1], xh.shape[2] - 3, node_mask)

        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
        # assert_mean_zero_with_mask(z_t[:, :, :3], node_mask)
        if noise_align:
            if pred_data:
                align_pos = get_align_position(z_t, xh)
            else:
                noise = get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask)

        # compute loss
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x = None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x = model(t, z_t, node_mask, edge_mask, noise_level=noise_level, cond_x=cond_x)
                    cond_x = cond_x.detach_()
            pred = model(t, z_t, node_mask, edge_mask, noise_level=noise_level, cond_x=cond_x)
        else:
            pred = model(t, z_t, node_mask, edge_mask, noise_level=noise_level)

        if pred_data:
            # data prediction loss
            losses_pos = torch.square(pred[:, :, :3] - align_pos)
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            atom_type_pred = pred[:, :, 3:]
            atom_type_tar = xh[:, :, 3:]
            losses_atom_types = torch.square(atom_type_pred - atom_type_tar).mean(dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
        else:
            # noise prediction loss
            losses_atom = torch.square(noise - pred)
            losses_pos = losses_atom[:, :, :3]
            losses_atom_types = losses_atom[:, :, 3:]
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_atom_types = torch.mean(losses_atom_types, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)

        if reduce_mean:
            losses_pos = losses_pos / n_nodes
            losses_atom_types = losses_atom_types / n_nodes

        losses = loss_weights[0] * losses_pos + loss_weights[1] * losses_atom_types
        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


def get_sde_2D_loss_fn(noise_scheduler, train, scaler, config):
    """The loss function for node features and edges. (2D graphs)"""
    device = config.device
    include_charges = config.model.include_fc_charge
    pred_data = config.model.pred_data
    reduce_mean = config.training.reduce_mean

    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    self_cond = config.model.self_cond

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data
        xh, edge_x, node_mask, edge_mask = process_batch_2D(batch, device, include_charges, scaler)

        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)
        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        noise = sample_gaussian_with_mask(xh.size(), xh.device, node_mask)
        edge_noise = sample_symmetric_edge_feature_noise(edge_x.shape[0], edge_x.shape[1], edge_x.shape[-1], edge_mask)

        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
        edge_z_t = expand_dims(alpha_t, edge_x.dim()) * edge_x + expand_dims(sigma_t, edge_noise.dim()) * edge_noise

        # compute loss
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x, cond_edge_x = None, None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x, cond_edge_x = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x)
                    cond_x, cond_edge_x = cond_x.detach_(), cond_edge_x.detach_()
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                    cond_x=cond_x, cond_edge_x=cond_edge_x)
        else:
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level)

        if pred_data:
            # data prediction loss
            losses_atom_types = torch.square(pred - xh).mean(dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.square(edge_x - edge_pred).mean(dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(xh.size(0), -1), dim=-1)
        else:
            # noise prediction loss
            losses_atom_types = torch.square(noise - pred)
            losses_atom_types = torch.mean(losses_atom_types, dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.square(edge_noise - edge_pred)
            losses_edge = torch.mean(losses_edge, dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(losses_edge.shape[0], -1), dim=-1)

        if reduce_mean:
            losses_atom_types = losses_atom_types / n_nodes
            losses_edge = losses_edge / (torch.sum(edge_mask.reshape(losses_edge.shape[0], -1), dim=-1) + 1e-8)

        losses = loss_weights[1] * losses_atom_types + loss_weights[2] * losses_edge

        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


def get_sde_graph_loss_fn(noise_scheduler, train, scaler, config, prop_norm=None):
    """The loss function for node features, positions and edge features"""

    device = config.device
    include_charges = config.model.include_fc_charge
    reduce_mean = config.training.reduce_mean
    noise_align = config.model.noise_align
    pred_data = config.model.pred_data
    loss_weights = config.model.loss_weights.split(',')
    loss_weights = [float(loss_weight) for loss_weight in loss_weights]
    self_cond = config.model.self_cond
    if self_cond:
        cond_process_fn = get_self_cond_fn(config)

    def loss_fn(model, batch):
        if train:
            model.train()
        else:
            model.eval()

        # process and normalize the batch data with edge data
        xh, edge_x, node_mask, edge_mask, context = process_edge_batch(batch, device, include_charges,
                                                                       scaler, prop_norm)
        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)

        # sample the noisy samples
        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1. - t_eps) + t_eps
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        noise = sample_combined_position_feature_noise(xh.shape[0], xh.shape[1], xh.shape[2] - 3, node_mask)
        edge_noise = sample_symmetric_edge_feature_noise(edge_x.shape[0], edge_x.shape[1], edge_x.shape[-1], edge_mask)

        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(sigma_t, noise.dim()) * noise
        edge_z_t = expand_dims(alpha_t, edge_x.dim()) * edge_x + expand_dims(sigma_t, edge_noise.dim()) * edge_noise

        # align position noise
        if noise_align:
            if pred_data:
                align_pos = get_align_position(z_t, xh)
            else:
                noise = get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask)
        else:
            align_pos = xh[:, :, :3]

        # compute loss
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        if self_cond:
            assert pred_data
            cond_x, cond_edge_x = None, None
            if random() < 0.5:
                with torch.no_grad():
                    cond_x, cond_edge_x = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                                cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
                    cond_x, cond_edge_x = cond_x.detach_(), cond_edge_x.detach_()
                    # post_process self_cond values
                    cond_x, cond_edge_x = cond_process_fn(cond_x, cond_edge_x)
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                    cond_x=cond_x, cond_edge_x=cond_edge_x, context=context)
        else:
            pred, edge_pred = model(t, z_t, node_mask, edge_mask, edge_x=edge_z_t, noise_level=noise_level,
                                    context=context)

        if pred_data:
            # data prediction loss
            losses_pos = torch.square(pred[:, :, :3] - align_pos)
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            atom_type_pred = pred[:, :, 3:]
            atom_type_tar = xh[:, :, 3:]
            losses_atom_types = torch.square(atom_type_pred - atom_type_tar).mean(dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.square(edge_x - edge_pred).mean(dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(xh.size(0), -1), dim=-1)
        else:
            # noise prediction loss
            losses_atom = torch.square(noise - pred)
            losses_edge = torch.square(edge_noise - edge_pred)
            losses_pos = losses_atom[:, :, :3]
            losses_atom_types = losses_atom[:, :, 3:]
            losses_pos = torch.mean(losses_pos, dim=-1)
            losses_atom_types = torch.mean(losses_atom_types, dim=-1)
            losses_edge = torch.mean(losses_edge, dim=-1)
            losses_pos = torch.sum(losses_pos, dim=-1)
            losses_atom_types = torch.sum(losses_atom_types, dim=-1)
            losses_edge = torch.sum(losses_edge.reshape(losses_edge.shape[0], -1), dim=-1)

        if reduce_mean:
            losses_pos = losses_pos / n_nodes
            losses_atom_types = losses_atom_types / n_nodes
            losses_edge = losses_edge / (torch.sum(edge_mask.reshape(losses_edge.shape[0], -1), dim=-1) + 1e-8)
        losses = loss_weights[0] * losses_pos + loss_weights[1] * losses_atom_types + loss_weights[2] * losses_edge

        # scale if predict data
        if pred_data:
            norm = torch.sqrt(alpha_t / sigma_t)
            losses = expand_dims(norm, losses.dim()) * losses

        return losses.mean()

    return loss_fn


#################### Utils Functions ####################

@torch.no_grad()
def get_align_noise(z_t, xh, alpha_t, sigma_t, noise, node_mask):
    pos_t = z_t[:, :, :3]
    pos_0 = xh[:, :, :3]

    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)

    noise_pos = (pos_t - expand_dims(alpha_t, align_pos_0.dim()) * align_pos_0) / expand_dims(sigma_t, pos_t.dim())
    noise[:, :, :3] = noise_pos
    return noise


@ torch.no_grad()
def get_align_position(z_t, xh):
    pos_t = z_t[:, :, :3]
    pos_0 = xh[:, :, :3]

    rotations = kabsch_batch(pos_t, pos_0)  # [batch_size, 3, 3]
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)

    return align_pos_0


@torch.no_grad()
def kabsch(coord_pred, coord_tar):
    A = coord_pred.transpose(0, 1) @ coord_tar  # [3, 3]
    U, S, Vt = torch.linalg.svd(A)
    corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=coord_pred.device))  # [3, 3]
    rotation = (U @ corr_mat) @ Vt
    return rotation


@torch.no_grad()
def kabsch_batch(coords_pred, coords_tar):
    """Batch version of Kabsch algorithm."""
    A = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar)
    U, S, Vt = torch.linalg.svd(A)
    sign_detA = torch.sign(torch.det(A))  # [batch_size]
    corr_mat_diag = torch.ones((A.size(0), U.size(-1)), device=A.device)  # [batch_size, 3]
    corr_mat_diag[:, -1] = sign_detA  # [batch_size, 3]
    corr_mat = torch.diag_embed(corr_mat_diag)  # [batch_size, 3, 3]
    rotation = torch.einsum("...ij, ...jk, ...kl -> ...il", U, corr_mat, Vt)  # [batch_size, 3, 3]

    return rotation


def process_batch(batch, device, include_charges, scaler):
    pos = batch['positions'].to(device)
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)
    edge_mask = batch['edge_mask'].to(device)
    atom_type = batch['one_hot'].to(device)
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)

    # scaler
    pos = remove_mean_with_mask(pos, node_mask)
    pos, atom_type, fc_charge = scaler(pos, atom_type, fc_charge, node_mask)

    # pack data
    xh = torch.cat([pos, atom_type, fc_charge], dim=2)

    return xh, node_mask, edge_mask


def process_batch_2D(batch, device, include_charges, scaler):
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)
    edge_mask = batch['edge_mask'].to(device)
    atom_type = batch['atom_one_hot'].to(device)
    edge_type = batch['edge_one_hot'].to(device)
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)
    pos = None

    # scaler
    # pos = remove_mean_with_mask(pos, node_mask)
    _, atom_type, fc_charge, edge_type = scaler(pos, atom_type, fc_charge, node_mask, edge_type, edge_mask)

    # pack data
    xh = torch.cat([atom_type, fc_charge], dim=2)
    return xh, edge_type, node_mask, edge_mask


@torch.no_grad()
def process_edge_batch(batch, device, include_charges, scaler, prop_norm):
    pos = batch['positions'].to(device)
    node_mask = batch['atom_mask'].to(device).unsqueeze(2)
    edge_mask = batch['edge_mask'].to(device)
    atom_type = batch['atom_one_hot'].to(device)
    edge_type = batch['edge_one_hot'].to(device)
    fc_charge = (batch['formal_charges'] if include_charges else torch.zeros(0)).to(device)

    # support context
    if 'context' in batch:
        context = batch['context'].to(device)
    else:
        context = None

    # scaler
    pos = remove_mean_with_mask(pos, node_mask)
    pos, atom_type, fc_charge, edge_type = scaler(pos, atom_type, fc_charge, node_mask, edge_type, edge_mask)

    if context is not None:
        for i, key in enumerate(prop_norm.keys()):
            context[:, i] = (context[:, i] - prop_norm[key]['mean']) / prop_norm[key]['mad']

    # pack data
    xh = torch.cat([pos, atom_type, fc_charge], dim=2)

    return xh, edge_type, node_mask, edge_mask, context
