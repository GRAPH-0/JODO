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


class MOSESDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['dataset_v1.csv']

    @property
    def processed_file_names(self) -> str:
        return ['data_moses.pt', 'split_dict_moses.pt']

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
        smile_list = list(input_df['SMILES'])

        split_series = input_df['SPLIT']
        train_idx = np.array(split_series[split_series == 'train'].index.to_list())
        valid_idx = np.array(split_series[split_series == 'test'].index.to_list())
        test_idx = np.array(split_series[split_series == 'test_scaffolds'].index.to_list())

        splits = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        torch.save(splits, self.processed_paths[1])

        types = {'C': 0, 'N': 1, 'S': 2, 'O': 3, 'F': 4, 'Cl': 5, 'Br': 6}
        bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

        data_list = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            atom_type_idx = []
            formal_charges = []
            for atom in mol.GetAtoms():
                atom_str = atom.GetSymbol()
                atom_type_idx.append(types[atom_str])
                formal_charges.append(atom.GetFormalCharge())

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
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
