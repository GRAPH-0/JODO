# from rdkit.Chem import PyMol
from rdkit.Chem import Draw
import os
import copy


def visualize_mols(mol_list, dir_path, config, check_valid=False):

    row = config.sampling.vis_row
    col = config.sampling.vis_col
    n_mol = row * col

    valid_mol_list = []
    if check_valid:
        for mol in mol_list:
            mol_copy = copy.deepcopy(mol)
            try:
                Chem.SantizeMol(mol_copy)
            except:
                continue
            smile = Chem.MolToSmiles(mol_copy)
            if smile is not None:
                valid_mol_list.append(mol)
            if len(valid_mol_list) >= n_mol:
                break
    else:
        valid_mol_list = mol_list[:n_mol]

    try:
        img = Draw.MolsToGridImage(valid_mol_list, subImgSize=(400, 400), molsPerRow=row)
        img.save(os.path.join(dir_path, 'mol.png'))
    except:
        pass
