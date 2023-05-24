import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torch_geometric.utils import dense_to_sparse
import torch_geometric.nn as pygnn
from torch_geometric.nn import Linear as Linear_pyg
from . import utils
from .layers import EdgeGateTransLayer
import math


class HybridMPBlock(nn.Module):
    """Local MPNN + fully-connected attention-based message passing layer. Inspired by GPSLayer."""

    def __init__(self, dim_h,
                 local_gnn_type='GINE', global_model_type='FullTrans_1', num_heads=8,
                 temb_dim=None, act=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        if act is None:
            self.act = nn.ReLU()
        else:
            self.act = act

        # time embedding
        if temb_dim is not None:
            self.t_node = nn.Linear(temb_dim, dim_h)
            self.t_edge = nn.Linear(temb_dim, dim_h)

        # local message-passing model
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'LocalTrans_1':
            self.local_model = EdgeGateTransLayer(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'FullTrans_1':
            self.self_attn = EdgeGateTransLayer(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")

        # Normalization for MPNN and Self-Attention representations.
        self.norm1_local = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)
        self.norm1_attn = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        # Feed Forward block -> node.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_node = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        # Feed Forward block -> edge.
        self.ff_linear3 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear4 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_edge = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

    def _ff_block_node(self, x):
        """Feed Forward block.
        """
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        """Feed Forward block.
        """
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, x, edge_index, dense_edge, dense_index, node_mask, adj_mask, temb=None):
        """
        Args:
            x: node feature [B*N, dim_h]
            edge_index: [2, edge_length]
            dense_edge: edge features in dense form [B, N, N, dim_h]
            dense_index: indices for valid edges [B, N, N, 1]
            node_mask: [B, N]
            adj_mask: [B, N, N, 1]
            temb: time conditional embedding [B, temb_dim]
        Returns:
            h
            edge
        """

        B, N, _, _ = dense_edge.shape
        h_in1 = x
        h_in2 = dense_edge

        if temb is not None:
            h_edge = (dense_edge + self.t_edge(self.act(temb))[:, None, None, :]) * adj_mask
            temb = temb.unsqueeze(1).repeat(1, N, 1)
            temb = temb.reshape(-1, temb.size(-1))
            h = (x + self.t_node(self.act(temb))) * node_mask.reshape(-1, 1)

        h_out_list = []
        # Local MPNN with edge attributes
        if self.local_model is not None:
            edge_attr = h_edge[dense_index]
            h_local = self.local_model(h, edge_index, edge_attr) * node_mask.reshape(-1, 1)
            h_local = h_in1 + self.dropout(h_local)
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention
        if self.self_attn is not None:
            if 'FullTrans' in self.global_model_type:
                # extract full connect edge_index and edge_attr
                dense_index_full = adj_mask.squeeze(-1).nonzero(as_tuple=True)
                edge_index_full, _ = dense_to_sparse(adj_mask.squeeze(-1))
                edge_attr_full = h_edge[dense_index_full]
                h_attn = self.self_attn(h, edge_index_full, edge_attr_full)
            else:
                raise ValueError(f"Unsupported global transformer layer")
            h_attn = h_in1 + self.dropout(h_attn)
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs
        assert len(h_out_list) > 0
        h = sum(h_out_list) * node_mask.reshape(-1, 1)
        h_dense = h.reshape(B, N, -1)
        h_edge = h_dense.unsqueeze(1) + h_dense.unsqueeze(2)

        # Feed Forward block
        h = h + self._ff_block_node(h)
        h = self.norm2_node(h) * node_mask.reshape(-1, 1)

        h_edge = h_in2 + self._ff_block_edge(h_edge)
        h_edge = self.norm2_edge(h_edge.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * adj_mask

        return h, h_edge


def conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, padding=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                     padding=padding)
    return conv


@torch.no_grad()
def get_rw_feat(k_step, dense_adj):
    """Compute k_step Random Walk for given dense adjacency matrix."""

    rw_list = []
    deg = dense_adj.sum(-1, keepdims=True)
    AD = dense_adj / (deg + 1e-8)
    rw_list.append(AD)

    for _ in range(k_step):
        rw = torch.bmm(rw_list[-1], AD)
        rw_list.append(rw)
    rw_map = torch.stack(rw_list[1:], dim=1)  # [B, k_step, N, N]

    rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)  # [B, k_step, N]
    rw_landing = rw_landing.permute(0, 2, 1)  # [B, N, rw_depth]

    # get the shortest path distance indices
    tmp_rw = rw_map.sort(dim=1)[0]
    spd_ind = (tmp_rw <= 0).sum(dim=1)  # [B, N, N]

    spd_onehot = torch.nn.functional.one_hot(spd_ind, num_classes=k_step+1).to(torch.float)
    spd_onehot = spd_onehot.permute(0, 3, 1, 2)  # [B, kstep, N, N]

    return rw_landing, spd_onehot


# from DDPM
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


@utils.register_model(name='CDGS')
class CDGS(nn.Module):
    """
    Graph Noise Prediction Model.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = nn.SiLU()

        # get input channels(data.num_channels), hidden channels(model.nf), number of blocks(model.num_res_blocks)
        self.nf = nf = config.model.nf
        self.num_gnn_layers = num_gnn_layers = config.model.n_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = 'positional'
        self.conditional = conditional = config.model.cond_time
        self.edge_th = -1.
        self.rw_depth = rw_depth = config.model.rw_depth
        assert config.data.centered

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 2))
            modules.append(nn.Linear(nf * 2, nf))

        atom_ch = config.data.atom_types
        bond_ch = config.model.edge_ch
        temb_dim = nf

        # project bond features
        # assert bond_ch == 2
        bond_se_ch = int(nf * 0.4)
        bond_type_ch = int(0.5 * (nf - bond_se_ch))
        modules.append(conv1x1(bond_ch - 1, bond_type_ch))
        modules.append(conv1x1(1, bond_type_ch))
        modules.append(conv1x1(rw_depth + 1, bond_se_ch))
        modules.append(nn.Linear(bond_se_ch + 2 * bond_type_ch, nf))

        # project atom features
        atom_se_ch = int(nf * 0.2)
        atom_type_ch = nf - 2 * atom_se_ch
        modules.append(nn.Linear(bond_ch, atom_se_ch))
        modules.append(nn.Linear(atom_ch, atom_type_ch))
        modules.append(nn.Linear(rw_depth, atom_se_ch))
        modules.append(nn.Linear(atom_type_ch + 2 * atom_se_ch, nf))
        self.x_ch = nf

        # gnn network
        cat_dim = (nf * 2) // num_gnn_layers
        for _ in range(num_gnn_layers):
            modules.append(HybridMPBlock(nf, 'GINE', 'FullTrans_1', config.model.n_heads,
                                         temb_dim=temb_dim, act=act, dropout=dropout, attn_dropout=dropout))
            modules.append(nn.Linear(nf, cat_dim))
            modules.append(nn.Linear(nf, cat_dim))

        # atom output
        modules.append(nn.Linear(cat_dim * num_gnn_layers + atom_type_ch, nf))
        modules.append(nn.Linear(nf, nf // 2))
        modules.append(nn.Linear(nf // 2, atom_ch))

        # bond output
        modules.append(conv1x1(cat_dim * num_gnn_layers + bond_type_ch, nf))
        modules.append(conv1x1(nf, nf // 2))
        modules.append(conv1x1(nf // 2, bond_ch - 1))

        # structure output
        modules.append(conv1x1(cat_dim * num_gnn_layers + bond_type_ch, nf))
        modules.append(conv1x1(nf, nf // 2))
        modules.append(conv1x1(nf // 2, 1))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, t, x, node_mask, edge_mask, context=None, *args, **kwargs):
        # original input: x, time_cond

        bond_feat = kwargs['edge_x'].permute(0, 3, 1, 2)  # [B, ch, N, N]
        atom_feat = x
        time_cond = t * 999

        bs, n_nodes, _ = x.shape

        atom_mask = node_mask.squeeze(-1)
        bond_mask = edge_mask.reshape(bs, 1, n_nodes, n_nodes)

        edge_exist = bond_feat[:, 0:1, :, :]
        edge_cate = bond_feat[:, 1:, :, :]

        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            temb = get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # rescale the input data to [-1, 1]
            atom_feat = atom_feat * 2. - 1.
            bond_feat = bond_feat * 2. - 1.

        # discretize dense adj
        with torch.no_grad():
            adj = edge_exist.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.] = 1.
            adj[adj < 0.] = 0.
            adj = adj * bond_mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        rw_landing, spd_onehot = get_rw_feat(self.rw_depth, adj)

        # construct edge feature [B, N, N, F]
        adj_mask = bond_mask.permute(0, 2, 3, 1)
        dense_cate = modules[m_idx](edge_cate).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_exist = modules[m_idx](edge_exist).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1) * adj_mask
        m_idx += 1
        dense_edge = modules[m_idx](torch.cat([dense_cate, dense_exist, dense_spd], dim=-1)) * adj_mask
        m_idx += 1

        # Use Degree as atom feature
        atom_degree = torch.sum(bond_feat, dim=-1).permute(0, 2, 1)  # [B, N, C]
        atom_degree = modules[m_idx](atom_degree)  # [B, N, nf]
        m_idx += 1
        atom_cate = modules[m_idx](atom_feat)
        m_idx += 1
        x_rwl = modules[m_idx](rw_landing)
        m_idx += 1
        x_atom = modules[m_idx](torch.cat([atom_degree, atom_cate, x_rwl], dim=-1))
        m_idx += 1
        h_atom = x_atom.reshape(-1, self.x_ch)
        # Dense to sparse node [BxN, -1]

        dense_index = adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj)
        h_dense_edge = dense_edge

        # Run GNN layers
        atom_hids = []
        bond_hids = []
        for _ in range(self.num_gnn_layers):
            h_atom, h_dense_edge = modules[m_idx](h_atom, edge_index, h_dense_edge, dense_index,
                                                  atom_mask, adj_mask, temb)
            m_idx += 1
            atom_hids.append(modules[m_idx](h_atom.reshape(x_atom.shape)))
            m_idx += 1
            bond_hids.append(modules[m_idx](h_dense_edge))
            m_idx += 1

        atom_hids = torch.cat(atom_hids, dim=-1)
        bond_hids = torch.cat(bond_hids, dim=-1)

        # Output
        atom_score = self.act(modules[m_idx](torch.cat([atom_cate, atom_hids], dim=-1))) \
                     * atom_mask.unsqueeze(-1)
        m_idx += 1
        atom_score = self.act(modules[m_idx](atom_score))
        m_idx += 1
        atom_score = modules[m_idx](atom_score)
        m_idx += 1

        bond_score = self.act(modules[m_idx](torch.cat([dense_cate, bond_hids], dim=-1).permute(0, 3, 1, 2))) \
                     * bond_mask
        m_idx += 1
        bond_score = self.act(modules[m_idx](bond_score))
        m_idx += 1
        bond_score = modules[m_idx](bond_score)
        m_idx += 1

        exist_score = self.act(modules[m_idx](torch.cat([dense_exist, bond_hids], dim=-1).permute(0, 3, 1, 2))) \
                      * bond_mask
        m_idx += 1
        exist_score = self.act(modules[m_idx](exist_score))
        m_idx += 1
        exist_score = modules[m_idx](exist_score)
        m_idx += 1

        # make score symmetric
        bond_score = torch.cat([exist_score, bond_score], dim=1)
        bond_score = (bond_score + bond_score.transpose(2, 3)) / 2.

        assert m_idx == len(modules)

        atom_score = atom_score * atom_mask.unsqueeze(-1)
        bond_score = bond_score * bond_mask

        return atom_score, bond_score.permute(0, 2, 3, 1)
