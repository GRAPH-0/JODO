from qm9_dataset import QM9Dataset
from geom_dataset import GeomDrugDataset
from zinc_dataset import ZincDataset
from moses_dataset import MOSESDataset
from tqdm import tqdm


# Obtain node number statistic

def node_num_hist(dataset):
    print('training graphs:', len(dataset))
    node_num_dict = dict()
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if type(data.num_atom) == int:
            num_atom = data.num_atom
        else:
            num_atom = data.num_atom.item()
        assert num_atom == data.atom_type.size(0)
        if num_atom in node_num_dict:
            node_num_dict[num_atom] += 1
        else:
            node_num_dict[num_atom] = 1
    print('training node:')
    str = '{'
    for key, value in sorted(node_num_dict.items(), key=lambda x: x[0]):
        str = str + "{}: {}, ".format(key, value)
    str = str[:-2] + '}'
    print(str)

    print('training max node:', max(node_num_dict.keys()))


if __name__ == "__main__":
    # QM9
    # print('QM9 Dataset')
    # qm9_root_path = 'data/QM9'
    # qm9_root_path = 'data/QM9'
    # dataset = QM9Dataset(qm9_root_path)

    # Geom
    # print('Geom Drug Dataset')
    # geom_root_path = 'data/geom'
    # geom_data_file = 'data_geom_drug_1.pt'
    # dataset = GeomDrugDataset(geom_root_path, geom_data_file)

    # Zinc250k
    # print('Zinc250k Dataset')
    # zinc_root_path = 'data/zinc250k'
    # dataset = ZincDataset(zinc_root_path)

    # MOSES
    print('MOSES Dataset')
    moses_root_path = 'data/MOSES'
    dataset = MOSESDataset(moses_root_path)

    split_idx = dataset.get_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    node_num_hist(train_dataset)
