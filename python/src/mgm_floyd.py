import torch

from utils.eval_metric import get_batch_pc_opt, get_single_pc_opt
from utils.eval_metric import get_batch_affinity, get_single_affinity


class Floyd(torch.nn.Module):
    def __init__(self, params):
        super(Floyd, self).__init__()
        self.params = params

    def forward(self, K, X, m, n):
        mat = mgm_floyd_solver(K=K,
                               X=X,
                               num_graph=m,
                               num_node=n,
                               const=self.params.const)
        return mat


def mgm_floyd_solver(K, X, num_graph, num_node, const):
    m, n = num_graph, num_node

    for k in range(m):
        pair_aff = get_batch_affinity(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - torch.eye(m).cuda() * pair_aff
        norm = torch.max(pair_aff)
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
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = get_single_affinity(X[i, j], K[i, j], norm=norm)
                con_ori = get_single_pc_opt(X, i, j)
                score_ori = aff_ori * (1 - const) + con_ori * const

                X_combo = torch.matmul(X[i, k], X[k, j])
                aff_combo = get_single_affinity(X_combo, K[i, j], norm=norm)
                con_combo = get_single_pc_opt(X, i, j, X_combo)
                score_combo = aff_combo * (1 - const) + con_combo * const

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose(0, 1)
    return X
