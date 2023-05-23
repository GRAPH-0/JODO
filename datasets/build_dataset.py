from torch_geometric.transforms import Compose, ToDevice
from .qm9_dataset import QM9Dataset
from .geom_dataset import GeomDrugDataset
from .zinc_dataset import ZincDataset
from .moses_dataset import MOSESDataset
from .datasets_config import get_dataset_info
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch


prop2idx = {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11}


def get_dataset(config, transform=True):
    """Create dataset for training and evaluation."""

    # Obtain dataset info
    dataset_info = get_dataset_info(config.data.info_name)

    # get transform
    if transform:
        name_transform = getattr(config.data, 'transform', 'EDM')
        if name_transform == 'Edge':
            transform = EdgeTransform(dataset_info['atom_encoder'].values(), config.data.bond_types)
        elif name_transform == 'EdgeCom':
            transform = EdgeComTransform(dataset_info['atom_encoder'].values(), config.data.include_aromatic)
        elif name_transform == 'EdgeComCond':
            prop2idx = dataset_info['prop2idx']
            transform = EdgeComCondTransform(dataset_info['atom_encoder'].values(), config.data.include_aromatic,
                                             prop2idx[config.cond_property])
        elif name_transform == 'EdgeComCondMulti':
            prop2idx = dataset_info['prop2idx']
            transform = EdgeComCondMultiTransform(dataset_info['atom_encoder'].values(), config.data.include_aromatic,
                                                  prop2idx[config.cond_property1], prop2idx[config.cond_property2])
        else:
            raise ValueError('Invalid data transform name')
    else:
        transform = None

    # Build up dataset
    if config.data.name == 'QM9':
        dataset = QM9Dataset(config.data.root, transform=transform)
    elif config.data.name == 'GeomDrug':
        dataset = GeomDrugDataset(config.data.root, config.data.processed_file, transform=transform)
    elif config.data.name == 'Zinc250k':
        dataset = ZincDataset(config.data.root, transform=transform)
    elif config.data.name == 'MOSES':
        dataset = MOSESDataset(config.data.root, transform=transform)
    else:
        raise ValueError('Undefined dataset name.')

    # Split dataset
    if 'cond' in config.exp_type:
        split_idx = dataset.get_cond_idx_split()
        first_train_dataset = dataset.index_select(split_idx['first_train'])
        second_train_dataset = dataset.index_select(split_idx['second_train'])
        val_dataset = dataset.index_select(split_idx['valid'])
        test_dataset = dataset.index_select(split_idx['test'])
        return first_train_dataset, second_train_dataset, val_dataset, test_dataset, dataset_info

    split_idx = dataset.get_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    val_dataset = dataset.index_select(split_idx['valid'])
    test_dataset = dataset.index_select(split_idx['test'])

    return train_dataset, val_dataset, test_dataset, dataset_info


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# setup dataloader
def get_dataloader(train_ds, val_ds, test_ds, config):
    # choose collate_fn
    collate_fn = eval(config.data.collate)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,
                              num_workers=config.data.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                            num_workers=config.data.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.training.eval_batch_size, shuffle=False,
                             num_workers=config.data.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# transform data

class EdgeTransform(object):
    """
    Transform data with node and edge features
    """
    def __init__(self, atom_type_list, edge_types):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.edge_type_list = torch.tensor([_ for _ in range(1, edge_types)])

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_one_hot = edge_type.unsqueeze(-1) == self.edge_type_list.unsqueeze(0)
        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_one_hot.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_one_hot.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_one_hot.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_one_hot.float())
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)
        data.edge_one_hot = dense_edge_one_hot

        return data


class EdgeComTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        # edge feature channel [edge_exist; bond_order; aromatic_exist]
        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot
        return data


class EdgeComCondTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx = property_idx

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

            data.property = properties[0, self.property_idx:self.property_idx+1] - \
                            torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            property = properties[0, self.property_idx:self.property_idx+1]
            data.property = property

        return data


class EdgeComCondMultiTransform(object):
    """
    Transform data with node and edge features. Compress single/double/triple bond types to one channel.
    Conditional property.

    Edge:
        0-th ch: exist edge or not
        1-th ch: 0, 1, 2, 3; other bonds, single, double, triple bonds
        2-th ch: aromatic bond or not
    """

    def __init__(self, atom_type_list, include_aromatic, property_idx1, property_idx2):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.include_aromatic = include_aromatic
        self.property_idx1 = property_idx1
        self.property_idx2 = property_idx2

    def __call__(self, data: Data):
        # add atom type one_hot
        atom_type = data.atom_type
        edge_type = data.edge_type

        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()

        # dense bond type [N_node, N_node, ch], single(1000), double(0100), triple(0010), aromatic(0001), none(0000)
        edge_bond = edge_type.clone()
        edge_bond[edge_bond == 4] = 0
        edge_bond = edge_bond / 3.
        edge_feat = [edge_bond]
        if self.include_aromatic:
            edge_aromatic = (edge_type == 4).float()
            edge_feat.append(edge_aromatic)
        edge_feat = torch.stack(edge_feat, dim=-1)

        edge_index = data.edge_index
        dense_shape = (data.num_nodes, data.num_nodes, edge_feat.size(-1))
        dense_edge_one_hot = torch.zeros((data.num_nodes**2, edge_feat.size(-1)), device=atom_type.device)

        idx1, idx2 = edge_index[0], edge_index[1]
        idx = idx1 * data.num_nodes + idx2
        idx = idx.unsqueeze(-1).expand(edge_feat.size())
        dense_edge_one_hot.scatter_add_(0, idx, edge_feat)
        dense_edge_one_hot = dense_edge_one_hot.reshape(dense_shape)

        edge_exist = (dense_edge_one_hot.sum(dim=-1, keepdim=True) != 0).float()
        dense_edge_one_hot = torch.cat([edge_exist, dense_edge_one_hot], dim=-1)
        data.edge_one_hot = dense_edge_one_hot

        properties = data.y
        prop_list = [self.property_idx1, self.property_idx2]
        property_data = []
        for prop_idx in prop_list:
            if prop_idx == 11:
                Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
                atom_types = data.atom_type
                atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))

                property_data.append(properties[0, prop_idx:prop_idx+1] - \
                                torch.sum((atom_counts * torch.tensor(Cv_atomref))))
            else:
                property = properties[0, prop_idx:prop_idx+1]
                property_data.append(property)
        data.property = torch.cat(property_data)

        return data


class PropClassifierTransform(object):
    """
        Transform data with node and edge features.
        Conditional property.

    """
    def __init__(self, atom_type_list, property_idx):
        super().__init__()
        self.atom_type_list = torch.tensor(list(atom_type_list))
        self.property_idx = property_idx

    def __call__(self, data: Data):
        data.charge = None
        atom_type = data.atom_type
        one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.one_hot = one_hot.float()
        if self.property_idx == 11:
            Cv_atomref = [2.981, 2.981, 2.981, 2.981, 2.981]
            atom_types = data.atom_type
            atom_counts = torch.bincount(atom_types, minlength=len(Cv_atomref))
            data.property = data.y[0, 11] - torch.sum((atom_counts * torch.tensor(Cv_atomref)))
        else:
            data.property = data.y[0, self.property_idx]

        return data


def pad_node_feature(x, pad_len):
    x_len, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)
        new_x[:x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_feature(x, pad_len):
    # x: [N_node, N_node, ch]
    x_len, _, x_dim = x.size()
    if x_len < pad_len:
        new_x = x.new_zeros([pad_len, pad_len, x_dim])
        new_x[:x_len, :x_len, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_node_mask(node_num, pad_len, dtype):
    node_mask = torch.zeros(pad_len, dtype=dtype)
    node_mask[:node_num] = 1.
    return node_mask.unsqueeze(0)


# collate function: padding with the max node

def collate_node(items):
    items = [(item.one_hot, item.pos, item.fc, item.num_atom) for item in items]
    one_hot, positions, formal_charges, num_atoms = zip(*items)
    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        formal_charges=formal_charges
    )


def collate_edge(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_edge_2D(items):
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.num_atom) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, num_atoms = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask
    )


def collate_cond(items):
    # collate_fn for the condition generation
    items = [(item.atom_one_hot, item.edge_one_hot, item.fc, item.pos, item.num_atom, item.property) for item in items]
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, property = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    # context property
    property = torch.stack(property, dim=0)

    # repack
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        context=property
    )


def collate_property_classifier(items):
    # add conds for the property iterations
    items = [(item.one_hot, item.pos, item.num_atom, item.property) for item in items]
    one_hot, positions, num_atoms, graph_properties = zip(*items)

    max_node_num = max(num_atoms)

    # padding features
    one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in one_hot])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])

    # atom mask
    node_mask = torch.cat([get_node_mask(i, max_node_num, one_hot.dtype) for i in num_atoms])

    # edge mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)

    property = torch.stack(graph_properties, dim=0)

    return dict(
        one_hot=one_hot,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        positions=positions,
        property=property
    )
