import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
import pickle
import json
from tqdm import tqdm
import copy


types = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
         'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
atom_charge_dict = dict()


def my_extract_conformers(args):
    rdkit_folder_path = os.path.join(args.data_dir, 'rdkit_folder')
    summary_path = os.path.join(rdkit_folder_path, 'summary_drugs.json')
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    save_path = [f'data_geom_drug_{args.conformations}.pt']

    # filter valid pickle path
    pickle_path_list = []
    ori_smiles_list = []

    for smiles, meta_mol in tqdm(summ.items()):
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        pickle_path_list.append(pickle_path)
        ori_smiles_list.append(smiles)

    print('Find %d drug molecules' % len(pickle_path_list))

    # get the lowest conformations
    data_list = []
    for i in tqdm(range(len(pickle_path_list))):

        with open(os.path.join(rdkit_folder_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)

        # sort conformers based on energy
        conformers = mol['conformers']
        all_energies = [conformer['totalenergy'] for conformer in conformers]
        all_energies = np.array(all_energies)
        argsort = np.argsort(all_energies)
        lowest_energies = argsort[:args.conformations]
        for id in lowest_energies:
            conf_data = conformers[id]
            data_list.append(rdmol_to_data(conf_data['rd_mol']))

    print(atom_charge_dict)
    print('Process %d drug molecule conformers' % len(data_list))
    torch.save(data_list, os.path.join(args.data_dir, save_path[0]))


def rdmol_to_data(mol):

    from torch_geometric.data import Data
    import rdkit
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT
    bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}

    N = mol.GetNumAtoms()
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    formal_charges = []

    for atom in mol.GetAtoms():
        atom_str = atom.GetSymbol()
        type_idx.append(types[atom_str])
        formal_charges.append(atom.GetFormalCharge())
        if formal_charges[-1] != 0:
            atom_charge = atom_str + str(formal_charges[-1])
            if atom_charge in atom_charge_dict:
                atom_charge_dict[atom_charge] += 1
            else:
                atom_charge_dict[atom_charge] = 1

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
    x = torch.tensor(type_idx)

    data = Data(atom_type=x, pos=pos, fc=torch.tensor(formal_charges),
                edge_index=edge_index, edge_type=edge_type, num_atom=N, rdmol=copy.deepcopy(mol))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=1,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='data/geom/')
    args = parser.parse_args()
    my_extract_conformers(args)
