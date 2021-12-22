import torch

from utils.eval_metric import get_batch_affinity, get_single_affinity
from utils.eval_metric import get_batch_pc_opt, get_single_pc_opt


class Floyd(torch.nn.Module):
    def __init__(self):
        super(Floyd, self).__init__()

    def forward(self, K, X, m, n, params, mode="pc"):
        if mode == "c":
            mat = mgm_floyd_solver(K=K,
                                   X=X,
                                   num_graph=m,
                                   num_node=n,
                                   const=params.const)
        elif mode == "pc":
            mat = mgm_floyd_fast_solver(K=K,
                                        X=X,
                                        num_graph=m,
                                        num_node=n,
                                        const=params.const)
        else:
            raise NotImplementedError
        return mat


def mgm_floyd_solver(K, X, num_graph, num_node, const):
    m, n = num_graph, num_node

    for k in range(m):
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, torch.mean(pair_aff).item(), torch.mean(get_batch_pc_opt(X)).item()
        # ))

        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                score_ori = get_single_affinity(X[i, j], K[i, j], norm=norm)
                X_combo = torch.matmul(X[i, k], X[k, j])
                score_combo = get_single_affinity(X_combo, K[i, j], norm=norm)

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose(0, 1)

    for k in range(m):
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)

        pair_con = get_batch_pc_opt(X)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = get_single_affinity(X[i, j], K[i, j], norm=norm)
                con_ori = get_single_pc_opt(X, i, j)
                # con_ori = torch.sqrt(pair_con[i, j])
                score_ori = aff_ori * (1 - const) + con_ori * const

                X_combo = torch.matmul(X[i, k], X[k, j])
                aff_combo = get_single_affinity(X_combo, K[i, j], norm=norm)
                con_combo = get_single_pc_opt(X, i, j, X_combo)
                # con_combo = torch.sqrt(pair_con[i, k] * pair_con[k, j])
                score_combo = aff_combo * (1 - const) + con_combo * const

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose(0, 1)
    return X


def mgm_floyd_fast_solver(K, X, num_graph, num_node, const):
    m, n = num_graph, num_node
    device = K.device

    mask1 = torch.arange(m).reshape(m, 1).repeat(1, m)
    mask2 = torch.arange(m).reshape(1, m).repeat(m, 1)
    mask = (mask1 < mask2).float().to(device)
    X_mask = mask.reshape(m, m, 1, 1)

    for k in range(m):
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, torch.mean(pair_aff).item(), torch.mean(get_batch_pc_opt(X)).item()
        # ))

        X1 = X[:, k].reshape(m, 1, n, n).repeat(1, m, 1, 1).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n).repeat(m, 1, 1, 1).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = torch.bmm(X1, X2).reshape(m, m, n, n)

        aff_ori = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n), norm=norm).reshape(m, m)
        aff_combo = get_batch_affinity(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n), norm=norm).reshape(m, m)

        score_ori = aff_ori
        score_combo = aff_combo

        upt = (score_ori < score_combo).float()
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose(0, 1).transpose(2, 3) * (1 - X_mask)

    for k in range(m):
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)

        pair_con = get_batch_pc_opt(X)

        X1 = X[:, k].reshape(m, 1, n, n).repeat(1, m, 1, 1).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n).repeat(m, 1, 1, 1).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = torch.bmm(X1, X2).reshape(m, m, n, n)

        aff_ori = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n), norm=norm).reshape(m, m)
        aff_combo = get_batch_affinity(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n), norm=norm).reshape(m, m)

        con_ori = torch.sqrt(pair_con)
        con1 = pair_con[:, k].reshape(m, 1).repeat(1, m)
        con2 = pair_con[k, :].reshape(1, m).repeat(m, 1)
        con_combo = torch.sqrt(con1 * con2)

        score_ori = aff_ori * (1 - const) + con_ori * const
        score_combo = aff_combo * (1 - const) + con_combo * const

        upt = (score_ori < score_combo).float()
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose(0, 1).transpose(2, 3) * (1 - X_mask)
    return X
