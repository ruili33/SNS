"""
Reference: https://github.com/XiaoxinHe/TAPE/blob/main/core/data_utils/load_cora.py
"""

import os
import sys
sys.path.append('../')

import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs

cora_mapping = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory"
}


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    num_classes = 7

    train_idx = []
    val_idx = []
    test_idx = []
    labels = data["y"]


    for i in range(num_classes):

        class_idx = torch.where(labels == i)[0]
        assert class_idx.size(0) >= 20, f"Not enough nodes for class {i}"
        
        permuted_idx = class_idx[torch.randperm(class_idx.size(0))]
        
        train_idx.extend(permuted_idx[:20].tolist())

    remaining_idx = list(set(range(data.num_nodes)) - set(train_idx))
    np.random.shuffle(remaining_idx)

    
    # test_idx = remaining_idx[00:1000]
    # val_idx = remaining_idx[1000:1500]
    val_idx = remaining_idx[:500]
    test_idx = remaining_idx[500:1500]
    

    # 确保我们有足够的节点来分配给验证集和测试集
    assert len(val_idx) == 500, "Not enough nodes for validation set"
    assert len(test_idx) == 1000, "Not enough nodes for test set"

    # 转换为tensor
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, data_citeid


# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    path = 'dataset/cora/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('dataset/cora/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1].replace(':', '_')
        pid_filename[pid] = fn

    path = 'dataset/cora/mccallum/cora/extractions/'

    text = {'title': [], 'abs': [], 'label': []}

    # Assuming path is given
    all_files = {f.lower().replace(':', '_'): f for f in os.listdir(path)}

    for pid in data_citeid:
        expected_fn = pid_filename[pid].lower()
        # fn = pid_filename[pid]
        if expected_fn in all_files.keys():
            real_fn = all_files[expected_fn]
            with open(path+real_fn) as f:
                lines = f.read().splitlines()
            
            if expected_fn in all_files:
                real_fn = all_files[expected_fn]

            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
            text['title'].append(ti)
            text['abs'].append(ab)
    
    for i in range(len(data.y)):
        text['label'].append(cora_mapping[data.y[i].item()])

    return data, text


if __name__ == "__main__":
    a=get_cora_casestudy()