import math
from itertools import product

import torch

from utils.config import cfg


def get_feature(n, points, adj):
    """
    :param n: points # of graph
    :param points: torch tensor, (n, 2)
    :param adj: torch tensor, (n, n)
    :return: edge feat, angle feat
    """
    points_1 = points.reshape(n, 1, 2).repeat(1, n, 1)
    points_2 = points.reshape(1, n, 2).repeat(n, 1, 1)
    edge_feat = torch.sqrt(torch.sum((points_1 - points_2) ** 2, dim=2))
    edge_feat = edge_feat / torch.max(edge_feat)
    angle_feat = torch.atan((points_1[:, :, 1] - points_2[:, :, 1]) / (points_1[:, :, 0] - points_2[:, :, 0] + 1e-8))
    angle_feat = 2 * angle_feat / math.pi

    return edge_feat, angle_feat


def get_pair_affinity(edge_feat_1, angle_feat_1, edge_feat_2, angle_feat_2, adj1, adj2):
    n1, n2 = edge_feat_1.shape[0], edge_feat_2.shape[0]
    assert n1 == angle_feat_1.shape[0] and n2 == angle_feat_2.shape[0]

    left_adj = adj1.reshape(n1, n1, 1, 1).repeat(1, 1, n2, n2)
    right_adj = adj2.reshape(1, 1, n2, n2).repeat(n1, n1, 1, 1)
    adj = left_adj * right_adj

    left_edge_feat = edge_feat_1.reshape(n1, n1, 1, 1, -1).repeat(1, 1, n2, n2, 1)
    right_edge_feat = edge_feat_2.reshape(1, 1, n2, n2, -1).repeat(n1, n1, 1, 1, 1)
    edge_weight = torch.sqrt(torch.sum((left_edge_feat - right_edge_feat) ** 2, dim=-1))

    left_angle_feat = angle_feat_1.reshape(n1, n1, 1, 1, -1).repeat(1, 1, n2, n2, 1)
    right_angle_feat = angle_feat_2.reshape(1, 1, n2, n2, -1).repeat(n1, n1, 1, 1, 1)
    angle_weight = torch.sqrt(torch.sum((left_angle_feat - right_angle_feat) ** 2, dim=-1))

    affinity = edge_weight * cfg.affinity.beta + angle_weight * (1 - cfg.affinity.beta)
    affinity = torch.exp(-affinity / cfg.affinity.scale_2D) * adj
    affinity = affinity.transpose(1, 2)

    return affinity


def get_K(n_points, points_list, adj_list):
    """
    :param n_points, list of int/tensor, (1), points # of graphs
    :param points_list, list of tensor, (n_i, 2) for each
    :param adj_list, list of tensor, (n_i, n_i) for each
    :return: affinity, torch tensor,
    """
    m = len(n_points)
    n_max = max(n_points)
    device = points_list[0].device
    affinity = torch.zeros(m, m, n_max, n_max, n_max, n_max).to(device)

    edge_feat_list = []
    angle_feat_list = []
    for n, points, adj in zip(n_points, points_list, adj_list):
        edge_feat, angle_feat = get_feature(n, points, adj)
        edge_feat_list.append(edge_feat)
        angle_feat_list.append(angle_feat)

    for i, j in product(range(m), range(m)):
        pair_affinity = get_pair_affinity(edge_feat_list[i],
                                          angle_feat_list[i],
                                          edge_feat_list[j],
                                          angle_feat_list[j],
                                          adj_list[i],
                                          adj_list[j])
        affinity[i, j] = pair_affinity

    affinity = affinity.permute(0, 1, 3, 2, 5, 4).reshape(m, m, n_max * n_max, n_max * n_max)
    return affinity
