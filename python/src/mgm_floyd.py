import torch

from utils.eval_metric import get_affinity_score, get_pairwise_consistency2


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

    # calculate score of initial matching X
    aff_score = get_affinity_score(X.reshape(-1, n, n), K.reshape(-1, n*n, n*n)).reshape(m, m)
    pair_con = get_pairwise_consistency2(X)
    score = pair_con * const + aff_score * (1 - const)
    print("aff_score", aff_score)
    print("pair_con", pair_con)
    print("score", score)

    for v in range(n):
        # calculate update matching Xu
        # Xu(i,j) = X(i,v) * X(v,j)
        X1 = X[:, v, :, :].repeat(1, m, 1, 1).reshape(-1, n, n)  # X1(i,j) = X(i,v)
        X2 = X[v, :, :, :].repeat(m, 1, 1, 1).reshape(-1, n, n)  # X2(i,j) = X(v,j)
        Xu = torch.bmm(X1, X2).reshape(m, m, n, n)  # Xu(i,j) = X(i,v) * X(v,j)

        # calculate affinity score of update matching
        aff_score_upd = get_affinity_score(Xu.reshape(-1, n, n), K.reshape(-1, n * n, n * n))
        aff_score_upd = aff_score_upd.reshape(m, m)

        # calculate pairwise consistency of update matching
        pc_tmp1 = pair_con[:, v].repeat(1, m)  # pc_tmp1(i,j) = pair_con(i,v)
        pc_tmp2 = pair_con[v, :].repeat(m, 1)  # pc_tmp2(i,j) = pair_con(v,j)
        pair_con_upd = torch.sqrt(pc_tmp1 * pc_tmp2)

        score_upd = pair_con_upd * const + aff_score_upd * (1 - const)

        # update
        idx = score_upd > score
        X = Xu * idx + X * (1 - idx)
        score = score_upd * idx + score * (1 - idx)

    return X
