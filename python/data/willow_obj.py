import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from data.base_dataset import BaseDataset
from utils.config import cfg
from utils.utils import lexico_iter

KPT_NAMES = {
    "Car": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Duck": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Face": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Motorbike": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Winebottle": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}


class WillowObject(BaseDataset):
    def __init__(self, sets, obj_resize, outlier=0):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(WillowObject, self).__init__()
        self.classes = cfg.WILLOW.CLASSES
        self.kpt_len = [cfg.WILLOW.KPT_LEN for _ in cfg.WILLOW.CLASSES]

        self.root_path = Path(cfg.WILLOW.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets == "train" or "test", "No match found for dataset {}".format(sets)
        self.split_offset = cfg.WILLOW.TRAIN_OFFSET
        self.train_len = cfg.WILLOW.TRAIN_NUM
        self.sets = sets
        self.outlier = outlier

        self.mat_list = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob("*.mat")]
            ori_len = len(cls_mat_list)
            assert ori_len > 0, "No data found for WILLOW Object Class. Is the dataset installed correctly?"
            if self.split_offset % ori_len + self.train_len <= ori_len:
                if sets == "train":
                    self.mat_list.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[: self.split_offset % ori_len]
                        + cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
            else:
                if sets == "train":
                    self.mat_list.append(
                        cls_mat_list[: (self.split_offset + self.train_len) % ori_len - ori_len]
                        + cls_mat_list[self.split_offset % ori_len:]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[
                        (self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len
                        ]
                    )

    def get_multi_cluster_samples(self, idx, ks=None, mode="all", cls_list=None, shuffle=True):
        assert ks is not None
        assert cls_list is not None
        anno_list = []
        perm_mat_list = []
        adj_mat_list = []
        node_label_list = []
        graph_indices_list = []
        for i, cls in enumerate(cls_list):
            if type(cls) == int:
                clss = self.classes[cls]
                cls_list[i] = clss
        if len(cls_list) < len(ks):
            for i in range(len(cls_list), len(ks)):
                cls_num = random.randrange(0, len(self.classes))
                cls = self.classes[cls_num]
                while cls in cls_list:
                    cls_num = random.randrange(0, len(self.classes))
                    cls = self.classes[cls_num]
                cls_list.append(cls)
        assert len(ks) == len(cls_list)
        for k, cls in zip(ks, cls_list):
            annos, perms, adjs, nodes, _ = self.get_k_samples(idx=idx, k=k, mode=mode, cls=cls, shuffle=shuffle)
            anno_list.extend(annos)
            perm_mat_list.extend(perms)
            adj_mat_list.extend(adjs)
            node_label_list.extend(nodes)
            # if type(cls) == str:
            #     cls = self.classes.index(cls)
            graph_indices_list.extend([cls for _ in range(k)])
        cnt = 0
        perm_mat_true_list = []
        for i, j in lexico_iter(graph_indices_list):
            if i == j:
                perm_mat_true_list.append(perm_mat_list[cnt])
                cnt += 1
            else:
                perm_mat_true_list.append(np.zeros((1, 1)))
        assert cnt == len(perm_mat_list)
        return anno_list, perm_mat_true_list, adj_mat_list, node_label_list, graph_indices_list

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True, num_iterations=200):
        """
        Randomly get a sample of k objects from VOC-Berkeley keypoints dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :param num_iterations: maximum number of iterations for sampling a datapoint
        :return: (k samples of data, k choose 2 groundtruth permutation matrices)
        """
        if idx is not None:
            raise NotImplementedError("No indexed sampling implemented for willow.")
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        if mode == "superset" and k == 2:
            anno_list, perm_mat = self.get_pair_superset(cls=cls, shuffle=shuffle, num_iterations=num_iterations)
            return anno_list, [perm_mat]

        anno_list = []
        cls_list = []
        for xml_name in random.sample(self.mat_list[cls], k):
            anno_dict = self.__get_anno_dict(xml_name, cls)
            if shuffle:
                random.shuffle(anno_dict["keypoints"])
            if len(anno_dict["keypoints"]) is not 10:
                return self.get_k_samples(idx, k, mode, cls, shuffle, num_iterations)
            anno_list.append(anno_dict)
            cls_list.append(cls)

        perm_mat_list = [
            np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32) for anno_pair in lexico_iter(anno_list)
        ]
        for n, ((s1, s2), (cls1, cls2)) in enumerate(zip(lexico_iter(anno_list), lexico_iter(cls_list))):
            row_list = []
            col_list = []
            if cls1 == cls2:
                for i, keypoint in enumerate(s1["keypoints"]):
                    for j, _keypoint in enumerate(s2["keypoints"]):
                        if keypoint["name"] == _keypoint["name"] and keypoint["name"] is not 10:
                            perm_mat_list[n][i, j] = 1
                            row_list.append(i)
                            col_list.append(j)
                            break
                if mode == "all":
                    pass
                elif mode == "rectangle" and k == 2:  # so far only implemented for k = 2
                    row_list.sort()
                    perm_mat_list[n] = perm_mat_list[n][row_list, :]
                    s1["keypoints"] = [s1["keypoints"][i] for i in row_list]
                    assert perm_mat_list[n].size == len(s1["keypoints"]) * len(s2["keypoints"])
                elif mode == "intersection" and k == 2:  # so far only implemented for k = 2
                    row_list.sort()
                    col_list.sort()
                    perm_mat_list[n] = perm_mat_list[n][row_list, :]
                    perm_mat_list[n] = perm_mat_list[n][:, col_list]
                    s1["keypoints"] = [s1["keypoints"][i] for i in row_list]
                    s2["keypoints"] = [s2["keypoints"][j] for j in col_list]
                else:
                    raise NotImplementedError(f"Unknown sampling strategy {mode}")

        return anno_list, perm_mat_list

    def get_pair_superset(self, cls=None, shuffle=True, num_iterations=200):
        """
        Randomly get a pair of objects from VOC-Berkeley keypoints dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = None

        anno_dict_1 = self.__get_anno_dict(random.sample(self.mat_list[cls], 1)[0], cls)
        if shuffle:
            random.shuffle(anno_dict_1["keypoints"])
        keypoints_1 = set([kp["name"] for kp in anno_dict_1["keypoints"]])

        for xml_name in random.sample(self.mat_list[cls], min(len(self.mat_list[cls]), num_iterations)):
            anno_dict_2 = self.__get_anno_dict(xml_name, cls)
            if shuffle:
                random.shuffle(anno_dict_2["keypoints"])
            keypoints_2 = set([kp["name"] for kp in anno_dict_2["keypoints"]])
            if keypoints_1.issubset(keypoints_2):
                anno_pair = [anno_dict_1, anno_dict_2]
                break

        if anno_pair is None:
            return self.get_pair_superset(cls, shuffle, num_iterations)

        perm_mat = np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]["keypoints"]):
            for j, _keypoint in enumerate(anno_pair[1]["keypoints"]):
                if keypoint["name"] == _keypoint["name"]:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break

        assert len(row_list) == len(anno_pair[0]["keypoints"])

        return anno_pair, perm_mat

    def get_pair(self, cls=None, shuffle=True):
        """
        Randomly get a pair of objects from WILLOW-object dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        for mat_name in random.sample(self.mat_list[cls], 2):
            anno_dict = self.__get_anno_dict(mat_name, cls)
            if shuffle:
                random.shuffle(anno_dict["keypoints"])
            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_["keypoints"]) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]["keypoints"]):
            for j, _keypoint in enumerate(anno_pair[1]["keypoints"]):
                if keypoint["name"] == _keypoint["name"]:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]["keypoints"] = [anno_pair[0]["keypoints"][i] for i in row_list]
        anno_pair[1]["keypoints"] = [anno_pair[1]["keypoints"][j] for j in col_list]

        return anno_pair, perm_mat

    def __get_anno_dict(self, mat_file, cls):
        """
        Get an annotation dict from .mat annotation
        """
        assert mat_file.exists(), "{} does not exist.".format(mat_file)

        img_name = mat_file.stem + ".png"
        img_file = mat_file.parent / img_name

        struct = sio.loadmat(mat_file.open("rb"))
        kpts = struct["pts_coord"]

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            obj = img.resize(self.obj_resize, resample=Image.BICUBIC)
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        keypoint_list = []
        for idx, keypoint in enumerate(np.split(kpts, kpts.shape[1], axis=1)):
            attr = {
                "name": idx,
                "x": float(keypoint[0]) * self.obj_resize[0] / w,
                "y": float(keypoint[1]) * self.obj_resize[1] / h
            }
            keypoint_list.append(attr)

        for i in range(self.outlier):
            attr = {
                "name": 10,
                "x": random.random() * self.obj_resize[0],
                "y": random.random() * self.obj_resize[1]
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict["image"] = obj
        anno_dict["keypoints"] = keypoint_list
        anno_dict["bounds"] = xmin, ymin, w, h
        anno_dict["ori_sizes"] = ori_sizes
        anno_dict["cls"] = cls

        return anno_dict
