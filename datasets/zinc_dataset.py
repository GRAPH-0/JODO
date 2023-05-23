import os
import os.path as osp
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import copy
from functools import lru_cache


def files_exist(files: List[str]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


class ZincDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['zinc250k_property.csv', 'valid_idx_zinc250k.json']

    @property
    def processed_file_names(self) -> str:
        return ['data_zinc250k.pt', 'split_dict_zinc250k.pt']

    def _download(self):
        if files_exist(self.processeded_paths):
            return

        if files_exist(self.raw_paths):
            return

        raise ValueError('Without raw files or processed files for Zinc250k Dataset.')

    def process(self):
        import pandas as pd
        import json
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        RDLogger.DisableLog('rdApp.*')

        input_df = pd.read_csv(self.raw_paths[0], sep=',', dtype='str')
        smile_list = list(input_df['smile'])

        with open(self.raw_paths[1]) as f:
            test_idx = np.array(json.load(f))
        train_idx = np.array(list(set(np.arange(len(smile_list))).difference(set(test_idx))))
        valid_num = int(len(smile_list) * 0.1)

        # random pick valid
        np.random.seed(0)
        data_perm = np.random.permutation(len(train_idx))
        valid_idx = train_idx[data_perm][:valid_num]
        splits = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(splits, self.processed_paths[1])

        # 6， 7， 8， 9， 15， 16， 17， 35， 53
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}  # 0 -> without edge

        data_list = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            Chem.Kekulize(mol)
            N = mol.GetNumAtoms()

            atom_type_idx = []
            formal_charges = []
            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                atom_type_idx.append(types[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                if bond.GetBondType() == BT.AROMATIC:
                    print('meet aromatic bond!')
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]

            data = Data(atom_type=torch.tensor(atom_type_idx), fc=torch.tensor(formal_charges), num_nodes=N,
                        edge_index=edge_index, edge_type=edge_type, num_atom=N, idx=i, rdmol=copy.deepcopy(mol))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)
        return data

    def get_idx_split(self):
        # load split idx for train, val, test
        return torch.load(self.processed_paths[1])
