qm9_with_h = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 4, 5: 5, 6: 9, 7: 16, 8: 49, 9: 124, 10: 362, 11: 807, 12: 1689, 13: 3060, 14: 5136,
                      15: 7796, 16: 10644, 17: 13025, 18: 13364, 19: 13832, 20: 9482, 21: 9970, 22: 3393, 23: 4848,
                      24: 539, 25: 1506, 26: 48, 27: 266, 29: 25},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                         'H1C-C1N-N1C', 'H1C-C1C-C1N'],
}


qm9_second_half = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 3, 5: 3, 6: 5, 7: 7, 8: 25, 9: 62, 10: 178, 11: 412, 12: 845, 13: 1541, 14: 2587,
                      15: 3865, 16: 5344, 17: 6461, 18: 6695, 19: 6944, 20: 4794, 21: 4962, 22: 1701, 23: 2380,
                      24: 267, 25: 754, 26: 17, 27: 132, 29: 15},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                         'H1C-C1N-N1C', 'H1C-C1C-C1N'],
    'prop2idx': {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11},
}

geom_with_h_1 = {
    'name': 'GeomDrug',
    'data_file': 'data_geom_drug_1.pt',
    'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
    'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],
    'train_n_nodes': {3: 2, 4: 1, 5: 2, 6: 1, 7: 2, 8: 6, 9: 12, 10: 14, 11: 18, 12: 39, 13: 51, 14: 60, 15: 86,
                      16: 108, 17: 145, 18: 257, 19: 295, 20: 355, 21: 528, 22: 744, 23: 1014, 24: 1390, 25: 1691,
                      26: 2216, 27: 2583, 28: 3163, 29: 3678, 30: 4367, 31: 4867, 32: 5423, 33: 6029, 34: 6558,
                      35: 7186, 36: 7596, 37: 7774, 38: 8275, 39: 8434, 40: 8434, 41: 8629, 42: 8920, 43: 8792,
                      44: 8882, 45: 8643, 46: 8438, 47: 8255, 48: 7883, 49: 7510, 50: 7224, 51: 6776, 52: 6315,
                      53: 5922, 54: 5485, 55: 5180, 56: 4742, 57: 4373, 58: 3919, 59: 3441, 60: 3085, 61: 2707,
                      62: 2390, 63: 1910, 64: 1806, 65: 1422, 66: 1125, 67: 953, 68: 824, 69: 602, 70: 587, 71: 456,
                      72: 359, 73: 287, 74: 260, 75: 210, 76: 191, 77: 136, 78: 125, 79: 120, 80: 95, 81: 75, 82: 62,
                      83: 54, 84: 56, 85: 47, 86: 47, 87: 46, 88: 41, 89: 24, 90: 18, 91: 23, 92: 25, 93: 17, 94: 25,
                      95: 18, 96: 16, 97: 19, 98: 9, 99: 17, 100: 16, 101: 6, 102: 9, 103: 5, 104: 10, 105: 5, 106: 10,
                      107: 19, 108: 11, 109: 4, 110: 9, 111: 15, 112: 6, 113: 8, 114: 3, 115: 2, 116: 5, 117: 14,
                      118: 20, 119: 7, 120: 8, 121: 3, 122: 1, 123: 13, 124: 15, 125: 7, 126: 10, 127: 7, 128: 4,
                      130: 2, 131: 1, 132: 4, 133: 4, 134: 10, 135: 8, 136: 7, 138: 10, 139: 3, 140: 21, 141: 4,
                      142: 10, 143: 3, 144: 4, 145: 16, 146: 3, 147: 5, 148: 16, 150: 10, 152: 1, 153: 3, 155: 4,
                      156: 3, 158: 2, 159: 1, 160: 2, 162: 1, 165: 1, 169: 1, 176: 1, 181: 1},
    'max_n_nodes': 181,
    'atom_fc_num': {'S1': 10931, 'N1': 33676, 'O-1': 31881, 'N-1': 60, 'P1': 243, 'C-1': 459, 'C1': 227, 'O1': 21,
                    'S3': 32, 'S-1': 5, 'B-1': 3, 'Br1': 3, 'H1': 9, 'S2': 8, 'I1': 1, 'Si1': 2, 'Cl-1': 1, 'I2': 3,
                    'Bi2': 1, 'P-1': 1, 'F-1': 1, 'N-2': 18, 'Cl1': 1},
    'colors_dic': ['#FFFFFF99',
                   'C2', 'C7', 'C0', 'C3', 'C1', 'C5',
                   'C6', 'C4', 'C8', 'C9', 'C10',
                   'C11', 'C12', 'C13', 'C14'],
    'radius_dic': [0.3, 0.6, 0.6, 0.6, 0.6,
                   0.6, 0.6, 0.6, 0.6, 0.6,
                   0.6, 0.6, 0.6, 0.6, 0.6,
                   0.6],
    'top_bond_sym': ['C1H', 'C12C', 'C1C', 'C1N', 'C12N', 'C1O', 'C2O', 'H1N'],
    'top_angle_sym': ['C12C-C12C', 'C1C-C1H', 'C12C-C1H', 'N1C-C1C', 'C1C-C1C', 'C1C-C12C', 'N1C-C1H', 'C1N-N1C'],
    'top_dihedral_sym': ['C12C-C12C-C1H', 'C12C-C12C-C12C', 'H1C-C1C-C1C', 'H1C-C1C-C1H', 'C1N-N1C-C1H', 'C1N-N1C-C1C',
                         'H1C-C1C-C12C', 'N1C-C1C-C1H']
}

zinc250k = {
    'name': 'Zinc250k',
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8},
    'atom_decoder': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
    'train_n_nodes': {6: 3, 7: 5, 8: 13, 9: 67, 10: 178, 11: 654, 12: 1053, 13: 1606, 14: 2532, 15: 3844, 16: 5695,
                      17: 7863, 18: 10489, 19: 13065, 20: 15906, 21: 18296, 22: 16825, 23: 18950, 24: 20907, 25: 20537,
                      26: 17331, 27: 14237, 28: 9057, 29: 6991, 30: 5478, 31: 4320, 32: 3375, 33: 2327, 34: 1553,
                      35: 925, 36: 358, 37: 126, 38: 2},
    'max_n_nodes': 38,
    'atom_fc_num': {'O-1': 24276, 'N1': 76787, 'N-1': 1539, 'S-1': 446, 'O1': 18, 'P1': 2, 'S1': 6, 'C-1': 3},
}

moses = {
    'name': 'MOSES',
    'atom_encoder': {'C': 0, 'N': 1, 'S': 2, 'O': 3, 'F': 4, 'Cl': 5, 'Br': 6},
    'atom_decoder': ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br'],
    'train_n_nodes': {8: 5, 9: 32, 10: 88, 11: 94, 12: 216, 13: 735, 14: 3689, 15: 5285, 16: 10943, 17: 37339,
                      18: 87694, 19: 176447, 20: 194878, 21: 202922, 22: 228280, 23: 237133, 24: 225489, 25: 144937,
                      26: 28454, 27: 3},
    'max_n_nodes': 27
}

dataset_info_dict = {
    'qm9_with_h': qm9_with_h,
    'geom_with_h_1': geom_with_h_1,
    'qm9_second_half': qm9_second_half,
    'zinc250k': zinc250k,
    'moses': moses
}


def get_dataset_info(info_name):
    return dataset_info_dict[info_name]