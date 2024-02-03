import torch
import os
import sys
sys.path.append('../')

import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
def get_raw_text_citeseer(SEED):
    dataset_name="citeseer"
    split= "fixed"
    dataset = torch.load(f"./dataset/citeseer/{dataset_name}_{split}_sbert.pt", map_location = 'cpu')
    text={}
    text['text']=dataset.raw_texts
    text['label']=dataset.category_names
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.


    node_id = np.arange(dataset.num_nodes)
    np.random.shuffle(node_id)

    num_nodes = dataset.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    num_classes = 6

    train_idx = []
    val_idx = []
    test_idx = []
    labels = dataset["y"]


    for i in range(num_classes):

        class_idx = torch.where(labels == i)[0]
        assert class_idx.size(0) >= 20, f"Not enough nodes for class {i}"
        
        permuted_idx = class_idx[torch.randperm(class_idx.size(0))]
        
        train_idx.extend(permuted_idx[:20].tolist())

    remaining_idx = list(set(range(dataset.num_nodes)) - set(train_idx))
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

    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask
    return dataset, text

