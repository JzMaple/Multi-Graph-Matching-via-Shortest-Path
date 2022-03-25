import copy
import json
import os
import time
import torch
import random

import numpy as np

from data.data_loader_multigraph import GMDataset, get_dataloader
from src.cao import CAO
from src.rrwm import RRWM
from src.mgm_floyd import Floyd
from utils.build_affinity import get_K
from utils.config import cfg
from utils.edge2adj import edge2adj
from utils.eval_metric import cal_accuracy, cal_affinity, cal_consistency
from utils.hungarian import hungarian
from utils.utils import update_params_from_cmdline, lexlist2tensor

rrwm_solver = RRWM()
cao_solver = CAO()
floyd_solver = Floyd()


def eval_test(mat, gt_mat, affinity, m, n):
    acc = cal_accuracy(mat, gt_mat, n)
    src = cal_affinity(mat, gt_mat, affinity, m, n)
    con = cal_consistency(mat, gt_mat, m, n)
    return acc, src, con


def update(acc, aff, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m, i_test):
    mat_accuracy[i_m][i_test] = acc
    mat_affinity[i_m][i_test] = aff
    mat_consistency[i_m][i_test] = con
    mat_time[i_m][i_test] = tim
    return mat_accuracy, mat_affinity, mat_consistency, mat_time


def offline_test(dataloader, device):
    ds = dataloader.dataset
    ds.set_num_graphs(cfg.TEST.num_graphs_in_matching_instance)
    classes = copy.deepcopy(ds.classes)

    method_list = ["rrwm", "cao", "cao-pc", "floyd", "floyd-pc"]
    n_method = 5

    for i, cls in enumerate(classes):
        print("Evaluation methods on {}:".format(cls))
        ds.set_cls(cls)
        mat_accuracy = torch.zeros(n_method, cfg.TEST.n_test)
        mat_affinity = torch.zeros(n_method, cfg.TEST.n_test)
        mat_consistency = torch.zeros(n_method, cfg.TEST.n_test)
        mat_time = torch.zeros(n_method, cfg.TEST.n_test)
        for i_test, inputs in enumerate(dataloader):
            points_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_list = [_.cuda() for _ in inputs["ns"]]
            graphs_list = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            adj_list = [edge2adj(graph.edge_index, n) for graph, n in zip(graphs_list, n_points_list)]
            K = get_K(n_points_list, points_list, adj_list)

            m = len(points_list)
            n = torch.max(torch.tensor(n_points_list))
            gt_mat = lexlist2tensor(perm_mat_list, m, torch.eye(n).to(device))

            # rrwm
            K_batch = K.reshape(-1, n * n, n * n)
            ns_src = torch.ones(m * m).int().to(device) * n
            ns_tgt = torch.ones(m * m).int().to(device) * n

            time_start = time.time()
            rrwm_mat = rrwm_solver(K_batch, n, ns_src, ns_tgt)
            rrwm_mat = rrwm_mat.reshape(-1, n, n).transpose(1, 2).contiguous()
            rrwm_mat = hungarian(rrwm_mat, ns_src, ns_tgt).reshape(m, m, n, n)
            time_end = time.time()
            base = time_end - time_start

            acc, src, con = eval_test(rrwm_mat, gt_mat, K, m, n)
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, base, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m=0, i_test=i_test
            )

            # CAO
            time_start = time.time()
            base_mat = copy.deepcopy(rrwm_mat)
            cao_mat = cao_solver(K, base_mat, m, n, cfg.cao_param, mode="c")
            time_end = time.time()
            acc, src, con = eval_test(cao_mat, gt_mat, K, m, n)
            tim = base + time_end - time_start
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m=1, i_test=i_test
            )

            # CAO fast
            # CAO pc requires much more memery ( O(m^3 n^4) ), which cannot be supported by GPU with 12GB.
            time_start = time.time()
            base_mat = copy.deepcopy(rrwm_mat)
            cao_fast_mat = cao_solver(K, base_mat, m, n, cfg.cao_fast_param, mode="pc")
            time_end = time.time()
            acc, src, con = eval_test(cao_fast_mat, gt_mat, K, m, n)
            tim = base + time_end - time_start
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m=2, i_test=i_test
            )

            # Floyd
            time_start = time.time()
            base_mat = copy.deepcopy(rrwm_mat)
            floyd_mat = floyd_solver(K, base_mat, m, n, cfg.floyd_param, mode="c")
            time_end = time.time()
            acc, src, con = eval_test(floyd_mat, gt_mat, K, m, n)
            tim = base + time_end - time_start
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m=3, i_test=i_test
            )

            # Floyd fast
            time_start = time.time()
            base_mat = copy.deepcopy(rrwm_mat)
            floyd_fast_mat = floyd_solver(K, base_mat, m, n, cfg.floyd_fast_param, mode="pc")
            time_end = time.time()
            acc, src, con = eval_test(floyd_fast_mat, gt_mat, K, m, n)
            tim = base + time_end - time_start
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m=4, i_test=i_test
            )

        for i_m in range(n_method):
            print("{:>10s}: accuracy is {:.4f}, affinity is {:.4f}, consistency is {:.4f}, time is {:.4f} ".format(
                method_list[i_m],
                torch.mean(mat_accuracy[i_m]).item(),
                torch.mean(mat_affinity[i_m]).item(),
                torch.mean(mat_consistency[i_m]).item(),
                torch.mean(mat_time[i_m]).item()
            ))
        print()


if __name__ == "__main__":
    cfg = update_params_from_cmdline(default_params=cfg)

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)

    seed = cfg.RANDOM_SEED
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_len = {"test": cfg.TEST.n_test * cfg.BATCH_SIZE}
    image_dataset = {
        "test": GMDataset(cfg.DATASET_NAME, sets="test", length=dataset_len["test"], obj_resize=(256, 256),
                          outlier=cfg.TEST.outlier)
    }
    dataloader = {"test": get_dataloader(image_dataset["test"], fix_seed=False)}

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    offline_test(dataloader["test"], device=device)
