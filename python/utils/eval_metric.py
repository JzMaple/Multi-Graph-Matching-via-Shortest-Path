import torch


def cal_accuracy(mat, gt_mat, n):
    acc = torch.mean(1 - torch.sum(torch.abs(mat - gt_mat), dim=[2, 3]) / 2 / n)
    return acc


def cal_consistency(mat, gt_mat, m, n):
    return None


def cal_affinity(mat, gt_mat, affinity, m, n):
    mat_batch = mat.reshape(-1, n, n)
    gt_mat_batch = gt_mat.reshape(-1, n, n)
    affinity_batch = affinity.reshape(-1, n * n, n * n)
    mat_as = get_affinity_score(mat_batch, affinity_batch)
    gt_as = get_affinity_score(gt_mat_batch, affinity_batch)
    return torch.mean(mat_as / (gt_as + 1e-8))


def _get_affinity_score(X, K):
    """
    calculate affinity score
    :param X: (b, n, n)
    :param K: (b, n*n, n*n)
    :return: affinity_score (b, 1, 1)
    """
    b, n, _ = X.size()
    vx = X.transpose(1, 2).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.transpose(1, 2)  # (b, 1, n*n)
    aff_score = torch.bmm(torch.bmm(vxt, K), vx)
    return aff_score


def get_affinity_score(X, K):
    aff_score = _get_affinity_score(X, K)
    aff_score = aff_score / torch.max(aff_score)
    return aff_score


def get_pairwise_consistency(X, Xu):
    """
    calculate pairwise consistency
    :param X: matching result permutation matrix (m, m, n, n)
    :param Xu: Xu(i,j,k) = X(i,k) * X(k,j) (m, m, m, n, n)
    :return: pair_consistency (m, m)
    """
    m, _, n, _ = X.size()

    X1 = Xu
    X2 = X.unsqueeze(2).repeat(1, 1, m, 1, 1)
    X3 = (X1 > X2).float()
    pair_con = 1 - torch.sum((torch.sum(X3, dim=(3, 4)) / torch.sum(X1, dim=(3, 4))), dim=2)
    return pair_con


def get_pairwise_consistency2(X):
    """
    calculate pairwise consistency
    :param X: matching result permutation matrix (m, m, n, n)
    :return: pair_consistency (m, m)
    """
    m, _, n, _ = X.size()
    X1 = X.unsqueeze(1).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1(i,j,k) = X(i,k)
    X2 = X.unsqueeze(0).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2(i,j,k) = X(k,j)
    Xu = torch.bmm(X1, X2).reshape(m, m, m, n, n)  # Xu(i,j,k) = X(i,k) * X(k,j)
    return get_pairwise_consistency(X, Xu)
