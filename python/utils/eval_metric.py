import torch


def cal_accuracy(mat, gt_mat, n):
    acc = torch.mean(1 - torch.sum(torch.abs(mat - gt_mat), dim=[2, 3]) / 2 / n)
    return acc


def cal_consistency(mat, gt_mat, m, n):
    return torch.mean(get_batch_pc_opt(mat))


def cal_affinity(X, X_gt, K, m, n):
    X_batch = X.reshape(-1, n, n)
    X_gt_batch = X_gt.reshape(-1, n, n)
    K_batch = K.reshape(-1, n * n, n * n)
    affinity = get_batch_affinity(X_batch, K_batch)
    affinity_gt = get_batch_affinity(X_gt_batch, K_batch)
    return torch.mean(affinity / (affinity_gt + 1e-8))


def get_batch_affinity(X, K, norm=1):
    """
    calculate affinity score
    :param X: (b, n, n)
    :param K: (b, n*n, n*n)
    :param norm: normalization term
    :return: affinity_score (b, 1, 1)
    """
    b, n, _ = X.size()
    vx = X.transpose(1, 2).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.transpose(1, 2)  # (b, 1, n*n)
    affinity = torch.bmm(torch.bmm(vxt, K), vx) / norm
    return affinity


def get_single_affinity(X, K, norm=1):
    """
    calculate affinity score
    :param X: (n, n)
    :param K: (n*n, n*n)
    :param norm: normalization term
    :return: affinity_score scale
    """
    n, _ = X.size()
    vx = X.transpose(0, 1).reshape(-1, 1)
    vxt = vx.transpose(0, 1)
    affinity = torch.matmul(torch.matmul(vxt, K), vx) / norm
    return affinity


def get_single_pc(X, i, j, Xij=None):
    """
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :param Xij: (n, n) matching
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.size()
    if Xij is None:
        Xij = X[i, j]
    pair_con = 0
    for k in range(m):
        X_combo = torch.matmul(X[i, k], X[k, j])
        pair_con += torch.sum(torch.abs(Xij - X_combo)) / (2 * n)
    return 1 - pair_con / m


def get_single_pc_opt(X, i, j, Xij=None):
    """
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.size()
    if Xij is None:
        Xij = X[i, j]
    X1 = X[i, :].reshape(-1, n, n)
    X2 = X[:, j].reshape(-1, n, n)
    X_combo = torch.bmm(X1, X2)
    pair_con = 1 - torch.sum(torch.abs(Xij - X_combo)) / (2 * n * m)
    return pair_con


def get_batch_pc(X):
    """
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    pair_con = torch.zeros(m, m).cuda()
    for i in range(m):
        for j in range(m):
            pair_con[i, j] = get_single_pc_opt(X, i, j)
    return pair_con


def get_batch_pc_opt(X):
    """
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m, _, n, _ = X.size()
    X1 = X.reshape(m, 1, m, n, n).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1[i, j, k] = X[i, k]
    X2 = X.reshape(1, m, m, n, n).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2[i, j, k] = X[k, j]
    X_combo = torch.bmm(X1, X2).reshape(m, m, m, n, n)
    X_ori = X.reshape(m, m, 1, n, n).repeat(1, 1, m, 1, 1)
    pair_con = 1 - torch.sum(torch.abs(X_combo - X_ori), dim=(2, 3, 4)) / (2 * n * m)
    return pair_con


if __name__ == "__main__":
    m = 5
    n = 10
    K = torch.rand(m, m, n * n, n * n).cuda()
    X = torch.zeros(m, m, n, n).cuda()
    for i in range(m):
        for j in range(m):
            if i == j:
                X[i, j] = torch.eye(n)
            if i > j:
                X[i, j] = X[j, i].transpose(0, 1)
            if i < j:
                perm = torch.randperm(n)
                X[i, j, range(n), perm] = 1.0

    affinity_single = torch.zeros(m, m).cuda()
    for i in range(m):
        for j in range(m):
            affinity_single[i, j] = get_single_affinity(X[i, j], K[i, j])

    K_batch = K.reshape(-1, n * n, n * n)
    X_batch = X.reshape(-1, n, n)
    affinity_batch = get_batch_affinity(X_batch, K_batch).reshape(m, m)

    assert torch.all(torch.abs(affinity_batch - affinity_single) < 1e-5)

    pair_con_ij = get_single_pc(X, 1, 2, X[2, 3])
    pair_con_ij_opt = get_single_pc_opt(X, 1, 2, X[2, 3])
    assert torch.all(torch.abs(pair_con_ij - pair_con_ij_opt) < 1e-5)

    pair_con_single = torch.zeros(m, m).cuda()
    for i in range(m):
        for j in range(m):
            pair_con_single[i, j] = get_single_pc(X, i, j)

    pair_con_batch = get_batch_pc(X)
    assert torch.all(torch.abs(pair_con_batch - pair_con_single) < 1e-5)

    pair_con_batch_opt = get_batch_pc_opt(X)
    assert torch.all(torch.abs(pair_con_batch - pair_con_single) < 1e-5)
