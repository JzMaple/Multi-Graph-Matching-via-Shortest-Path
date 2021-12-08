import copy
import json
import os
import time

import torch

from data.data_loader_multigraph import GMDataset, get_dataloader
from src.rrwm import RRWM
from utils.config import cfg
from utils.edge2adj import edge2adj
from utils.eval_metric import cal_accuracy, cal_affinity, cal_consistency
from utils.get_affinity import get_affinity
from utils.hungarian import hungarian
from utils.utils import update_params_from_cmdline, lexlist2tensor


def eval_test(mat, gt_mat, affinity, m, n):
    acc = cal_accuracy(mat, gt_mat, n)
    src = cal_affinity(mat, gt_mat, affinity, m, n)
    con = cal_consistency(mat, gt_mat, m, n)
    return acc, src, con


def update(acc, aff, con, tim, mat_accuracy, mat_affinity, mat_consistency, mat_time, i_m):
    mat_accuracy[i_m] += acc
    mat_affinity[i_m] += aff
    # mat_consistency[i_m] += con
    mat_time[i_m] += tim
    return mat_accuracy, mat_affinity, mat_consistency, mat_time


def offline_test(dataloader, device):
    rrwm_solver = RRWM()
    ds = dataloader.dataset
    ds.set_num_graphs(cfg.TEST.num_graphs_in_matching_instance)
    classes = copy.deepcopy(ds.classes)
    method_list = ["rrwm", "rrwm-old"]
    n_method = 2
    for i, cls in enumerate(classes):
        print("Evaluation methods on {}:".format(cls))
        ds.set_cls(cls)
        mat_accuracy = torch.zeros(n_method)
        mat_affinity = torch.zeros(n_method)
        mat_consistency = torch.zeros(n_method)
        mat_time = torch.zeros(n_method)
        cnt = 0
        for inputs in dataloader:
            points_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_list = [_.cuda() for _ in inputs["ns"]]
            graphs_list = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            adj_list = [edge2adj(graph.edge_index, n) for graph, n in zip(graphs_list, n_points_list)]
            affinity = get_affinity(n_points_list, points_list, adj_list)

            m = len(points_list)
            n = torch.max(torch.tensor(n_points_list))
            gt_mat = lexlist2tensor(perm_mat_list, m, torch.eye(n).to(device))

            # rrwm
            affinity_batch = affinity.reshape(-1, n * n, n * n)
            ns_src = torch.ones(m * m).int().to(device) * n
            ns_tgt = torch.ones(m * m).int().to(device) * n

            time_start = time.time()
            rrwm_mat = rrwm_solver(affinity_batch, n, ns_src, ns_tgt)
            rrwm_mat = rrwm_mat.reshape(-1, n, n).transpose(1, 2).contiguous()
            rrwm_mat = hungarian(rrwm_mat, ns_src, ns_tgt).reshape(m, m, n, n)
            time_end = time.time()

            acc, src, con = eval_test(rrwm_mat, gt_mat, affinity, m, n)
            mat_accuracy, mat_affinity, mat_consistency, mat_time = update(
                acc, src, con, time_end - time_start, mat_accuracy, mat_affinity, mat_consistency, mat_time, 0
            )

            cnt += 1

        for i_m in range(n_method):
            print("{:>10s}: accuracy is {:.4f}, affinity is {:.4f}, consistency is {:.4f}, time is {:.4f} ".format(
                method_list[i_m], mat_accuracy[i_m].item() / cnt, mat_affinity[i_m].item() / cnt,
                mat_consistency[i_m].item() / cnt, mat_time[i_m].item() / cnt
            ))
        print()


if __name__ == "__main__":
    cfg = update_params_from_cmdline(default_params=cfg)

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {"test": cfg.TEST.n_test * cfg.BATCH_SIZE}
    image_dataset = {
        "test": GMDataset(cfg.DATASET_NAME, sets="test", length=dataset_len["test"], obj_resize=(256, 256),
                          outlier=cfg.TEST.outlier)
    }
    dataloader = {"test": get_dataloader(image_dataset["test"], fix_seed=False)}

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    offline_test(dataloader["test"], device=device)
