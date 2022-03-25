import numpy as np
import torch

from utils.eval_metric import get_single_affinity, get_batch_affinity
from utils.eval_metric import get_single_pc_opt, get_batch_pc_opt


class CAO(torch.nn.Module):
    def __init__(self):
        super(CAO, self).__init__()

    def forward(self, K, X, m, n, params, mode="c"):
        if mode == "pc":
            mat = cao_fast_solver(
                K=K,
                X=X,
                num_graph=m,
                num_node=n,
                iter_max=params.iter_max,
                const_init=params.const_init,
                const_step=params.const_step,
                const_max=params.const_max,
                iter_boost=params.iter_boost
            )
        elif mode == "c":
            mat = cao_solver(
                K=K,
                X=X,
                num_graph=m,
                num_node=n,
                iter_max=params.iter_max,
                const_init=params.const_init,
                const_step=params.const_step,
                const_max=params.const_max,
                iter_boost=params.iter_boost
            )
        else:
            raise NotImplementedError
        return mat


def cao_solver(K, X, num_graph, num_node, iter_max=6, const_init=0.3, const_step=1.1, const_max=1.0, iter_boost=2):
    """
    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X, initial matching, (m, m, n, n)
    :param num_graph: number of nodes, int
    :param num_node: number of graphs, int
    :param iter_max: parameter
    :param const_init: parameter
    :param const_step: parameter
    :param const_max: parameter
    :param iter_boost: parameter
    :param mode: c or pc
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    const = const_init
    for iter in range(iter_max):
        if iter >= iter_boost:
            const = np.min([const * const_step, const_max])
        # pair_con = get_batch_pc_opt(X)
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = get_single_affinity(X[i, j], K[i, j], norm=norm)
                con_ori = get_single_pc_opt(X, i, j)
                # con_ori = torch.sqrt(pair_con[i, j])
                if iter < iter_boost:
                    score_ori = aff_ori
                else:
                    score_ori = aff_ori * (1 - const) + con_ori * const
                X_upt = X[i, j]
                for k in range(m):
                    X_combo = torch.matmul(X[i, k], X[k, j])
                    aff_combo = get_single_affinity(X_combo, K[i, j], norm=norm)
                    con_combo = get_single_pc_opt(X, i, j, X_combo)
                    # con_combo = torch.sqrt(pair_con[i, k] * pair_con[k, j])
                    if iter < iter_boost:
                        score_combo = aff_combo
                    else:
                        score_combo = aff_combo * (1 - const) + con_combo * const
                    if score_combo > score_ori:
                        X_upt = X_combo
                X[i, j] = X_upt
                X[j, i] = X_upt.transpose(0, 1)
    return X


def cao_fast_solver(K, X, num_graph, num_node, iter_max=6, const_init=0.3, const_step=1.1, const_max=1.0, iter_boost=2):
    """
    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X, initial matching, (m, m, n, n)
    :param num_graph: number of nodes, int
    :param num_node: number of graphs, int
    :param iter_max: parameter
    :param const_init: parameter
    :param const_step: parameter
    :param const_max: parameter
    :param iter_boost: parameter
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    const = const_init

    device = K.device
    mask1 = torch.arange(m).reshape(m, 1).repeat(1, m).to(device)
    mask2 = torch.arange(m).reshape(1, m).repeat(m, 1).to(device)
    mask = (mask1 < mask2).float()
    X_mask = mask.reshape(m, m, 1, 1)

    for iter in range(iter_max):
        if iter >= iter_boost:
            const = np.min([const * const_step, const_max])

        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)

        X1 = X.reshape(m, 1, m, n, n).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1[i,j,k] = X[i,k]
        X2 = X.reshape(1, m, m, n, n).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2[i,j,k] = X[k,j]
        X_combo = torch.bmm(X1, X2).reshape(m, m, m, n, n) # X_combo[i,j,k] = X[i, k] * X[k, j]

        aff_ori = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n), norm).reshape(m, m)
        pair_con = get_batch_pc_opt(X)
        con_ori = torch.sqrt(pair_con)

        K_repeat = K.reshape(m, m, 1, n * n, n * n).repeat(1, 1, m, 1, 1).reshape(-1, n * n, n * n)
        aff_combo = get_batch_affinity(X_combo.reshape(-1, n, n), K_repeat, norm).reshape(m, m, m)
        con1 = pair_con.reshape(m, 1, m).repeat(1, m, 1)  # con1[i,j,k] = pair_con[i,k]
        con2 = pair_con.reshape(1, m, m).repeat(m, 1, 1).transpose(1, 2)  # con2[i,j,k] = pair_con[j,k]
        con_combo = torch.sqrt(con1 * con2)

        if iter < iter_boost:
            score_ori = aff_ori
            score_combo = aff_combo
        else:
            score_ori = aff_ori * (1 - const) + con_ori * const
            score_combo = aff_combo * (1 - const) + con_combo * const

        score_combo, idx = torch.max(score_combo, dim=-1)

        assert torch.all(score_combo >= score_ori), torch.min(score_combo - score_ori)
        X_upt = X_combo[mask1, mask2, idx, :, :]
        X = X_upt * X_mask + X_upt.transpose(0, 1).transpose(2, 3) * X_mask.transpose(0, 1) + X * (1 - X_mask - X_mask.transpose(0, 1))
        assert torch.all(X.transpose(0, 1).transpose(2, 3) == X)

    return X
