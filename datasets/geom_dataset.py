import os.path as osp
import copy
import numpy as np
from functools import lru_cache
from collections.abc import Sequence
from typing import Callable, Optional
import torch
from torch_geometric.data import Dataset


class GeomDrugDataset(Dataset):
    def __init__(self, root: str, data_file: str, transform: Optional[Callable] = None):
        super(GeomDrugDataset, self).__init__()
        self.root = root
        self.data_file = data_file
        self.data = torch.load(osp.join(root, data_file))
        self.transform = transform

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = copy.copy(self.data[self.indices()[idx]])
        data = data if self.transform is None else self.transform(data)
        data.idx = idx
        return data

    def len(self):
        return len(self.data)

    def get_idx_split(self):
        # load split idx for train, val, test
        split_path = osp.join(self.root, self.data_file.replace('data', 'split_dict'))
        if osp.exists(split_path):
            print('Loading existing split data.')
            return torch.load(split_path)

        data_num = len(self.indices())

        val_proportion = 0.1
        test_proportion = 0.1

        valid_index = int(val_proportion * data_num)
        test_index = valid_index + int(test_proportion * data_num)

        # Generate random permutation
        data_perm = np.random.permutation(data_num)

        valid, test, train = np.split(data_perm, [valid_index, test_index])

        train = np.array(self.indices())[train]
        valid = np.array(self.indices())[valid]
        test = np.array(self.indices())[test]

        splits = {'train': train, 'valid': valid, 'test': test}
        torch.save(splits, split_path)
        return splits


if __name__ == '__main__':
    print('Test Geom Drug Dataset')
    root_path = 'data/geom'
    data_file = 'data_geom_drug_1.pt'
    dataset = GeomDrugDataset(root=root_path, data_file=data_file)

    print(len(dataset))
    split_idx = dataset.get_idx_split()
    train_dataset = dataset.index_select(split_idx['train'])
    val_dataset = dataset.index_select(split_idx['valid'])
    test_dataset = dataset.index_select(split_idx['test'])
    print(train_dataset[0])
    print(val_dataset[0])
    print(test_dataset[0])
