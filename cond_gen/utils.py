import torch

@torch.no_grad()
def compute_mean_mad_from_dataset(dataset, prop_idx):
    if prop_idx == 11:
        values = torch.cat([dataset.sub_Cv_thermo(dataset[i]).reshape(1) for i in range(len(dataset))])
    else:
        values = torch.cat([dataset[i].y[0, prop_idx].reshape(1) for i in range(len(dataset))])
    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    return mean, mad


def get_adj_matrix_fn():
    edges_dic = {}

    def get_adj_matrix(n_nodes, batch_size, device):
        if n_nodes in edges_dic:
            edges_dic_b = edges_dic[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            edges_dic[n_nodes] = {}
            return get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges

    return get_adj_matrix
