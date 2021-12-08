import torch

def edge2adj(edge_index, n):
    """
    :param edge_index: tensor (2, n), edge index
    :param n: int/tensor, # of points
    :return: adj
    """
    device = edge_index.device
    adj = torch.zeros(n, n).to(device)
    adj[edge_index[0], edge_index[1]] = 1
    return adj
