import torch

from utils.eval_metric import get_affinity_score, get_pairwise_consistency


class CAO(torch.nn.Module):
    def __init__(self, params):
        super(CAO, self).__init__()
        self.params = params

    def forward(self, K, X, m, n):
        mat = cao_solver(K=K,
                         X=X,
                         num_graph=m,
                         num_node=n,
                         iter_max=self.params.itermax,
                         const_init=self.params.const_init,
                         const_step=self.params.const_step,
                         const_max=self.params.const_max,
                         iter_boost=self.params.iter_boost)
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
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    const = const_init

    for i in range(iter_max):
        # calculate update matching Xu
        X1 = X.unsqueeze(1).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1(i,j,k) = X(i,k)
        X2 = X.unsqueeze(0).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2(i,j,k) = X(k,j)
        Xu = torch.bmm(X1, X2).reshape(m, m, m, n, n)  # Xu(i,j,k) = X(i,k) * X(k,j)

        # calculate affinity score of update matching
        Ku = K.unsqueeze(2).repeat(1, 1, m, 1, 1)  # Ku(i,j,k) = K(i,j) (m, m, m, n*n, n*n)
        aff_score_upd = get_affinity_score(Xu.reshape(-1, n, n), Ku.reshape(-1, n * n, n * n))
        aff_score_upd = aff_score_upd.reshape(m, m, m)

        if i >= iter_boost:
            # calculate pairwise consistency of update matching
            # pair_con_upd(i,j,k) = sqrt(pair_con(i,k) * pair_con(j,k))
            pair_con = get_pairwise_consistency(X, Xu)
            print("pair_con", pair_con)
            # assert torch.sum(pair_con - pair_con.transpose(0, 1)) == 0
            pc_tmp1 = pair_con.unsqueeze(1).repeat(1, m, 1)  # pc_tmp1(i,j,k) = pair_con(i,k)
            pc_tmp2 = pair_con.transpose(0, 1).unsqueeze(0).repeat(m, 1, 1)  # pc_tmp2(i,j,k) = pair_con(k,j)
            pair_con_upd = torch.sqrt(pc_tmp1 * pc_tmp2)

            score_upd = pair_con_upd * const + aff_score_upd * (1 - const)
        else:
            score_upd = aff_score_upd

        # update matching results
        idx = torch.argmax(score_upd, dim=2).reshape(-1)
        X = Xu.reshape(-1, m, n, n)[torch.arange(m * m), idx].reshape(m, m, n, n)  # X(i,j) = Xu(i,j, idx(i,j))
        # assert torch.sum(torch.abs(X.transpose(0, 1).transpose(2, 3) - X)) == 0

        # update const
        if i >= iter_boost:
            const = min(const_step * const, const_max)

    return X
