import torch
import os
import logging
import numpy as np


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=True)  # change strict to False?
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def get_data_scaler(config):
    """Data normalizer"""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if centered:
            atom_type = atom_type * 2. - 1.

        if pos is not None:
            pos = pos / pos_norm * node_mask
        atom_type = atom_type / atom_type_norm * node_mask
        fc_charge = fc_charge / fc_charge_norm * node_mask

        if edge_type is not None:
            if centered:
                edge_type = edge_type * 2. - 1.
            edge_type = edge_type / edge_norm
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return scale_fn


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def inverse_scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if pos is not None:
            pos = pos * pos_norm * node_mask
        atom_type = atom_type * atom_type_norm
        fc_charge = fc_charge * fc_charge_norm * node_mask
        if centered:
            atom_type = (atom_type + 1.) / 2. * node_mask

        if edge_type is not None:
            edge_type = edge_type * edge_norm
            if centered:
                edge_type = (edge_type + 1.) / 2.
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return inverse_scale_fn


def get_self_cond_fn(config):
    # To simplify: directly return

    process_type = config.model.self_cond_type  # 'ori', 'clamp'
    compress_edge = config.data.compress_edge
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    atom_type_scale = np.array([0., 1.])
    fc_scale = np.array(config.data.fc_scale)
    edge_type_scale = np.array([0., 1.])
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors
    _, atom_type_norm, fc_norm, edge_norm = normalize_factors

    # get the value scale
    centered = config.data.centered
    if centered:
        atom_type_scale = atom_type_scale * 2. - 1.
        edge_type_scale = edge_type_scale * 2. - 1.
    atom_type_scale = atom_type_scale / atom_type_norm
    fc_scale = fc_scale / fc_norm
    edge_type_scale = edge_type_scale / edge_norm

    def process_self_cond(cond_x, cond_edge_x):
        if process_type == 'ori':
            return cond_x, cond_edge_x
        elif process_type == 'clamp':
            atom_x = cond_x[:, :, 3:3+atom_types]
            atom_x = atom_x.clamp(atom_type_scale[0], atom_type_scale[1])
            cond_x[:, :, 3:3+atom_types] = atom_x
            if include_fc:
                atom_fc = cond_x[:, :, -1:]
                atom_fc = atom_fc.clamp(fc_scale[0], fc_scale[1])
                cond_x[:, :, -1:] = atom_fc
            cond_edge_x = cond_edge_x.clamp(edge_type_scale[0], edge_type_scale[1])
            return cond_x, cond_edge_x
        else:
            raise ValueError("Self-condition data process error.")

    return process_self_cond


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
