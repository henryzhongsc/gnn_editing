import logging
logger = logging.getLogger("main")

from typing import Tuple, Union, Optional
import time
import numpy as np
import pandas as pd
import os
import copy
import torch
import scipy.io as sio
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    # train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    # val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if train_per_class < 1:
            train_per_class = int(train_per_class * idx.size(0) / num_splits)
        if val_per_class < 1:
            val_per_class = int(val_per_class * idx.size(0) / num_splits)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]
        train_idx = idx[:train_per_class]
        # train_mask.scatter_(0, train_idx, True)
        train_mask[train_idx] = True
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        # val_mask.scatter_(0, val_idx, True)
        val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(),
                        T.RandomNodeSplit('train_rest', num_val=500, num_test=500)])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(f'{root}/WIKICS', transform=None)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Coauthor(f'{root}/Coauthor', name, transform=None)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = Amazon(f'{root}/Amazon', name, transform=None)
    data = dataset[0]
    torch.manual_seed(12345)
    data.train_mask, data.val_mask, data.test_mask = gen_masks(
        data.y, 20, 30, 20)
    return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB', transform=None)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB', transform=None)
    data = dataset[0]
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str) -> Tuple[Data, int, int]:
    dataset = Yelp(f'{root}/YELP', transform=None)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr', transform=None)
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2', transform=None)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes



def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train')
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes

def get_yelpchi(root: str, name: str) -> Tuple[Data, int, int]:
    dataset = sio.loadmat(root, verify_compressed_data_integrity=False)
    num_features = dataset['features'].shape[1]
    num_classes = len(np.unique(dataset['label']))
    edge_index = torch.tensor([dataset['homo'].nonzero()[0], dataset['homo'].nonzero()[1]], dtype=torch.long)
    x = torch.tensor(pd.DataFrame.sparse.from_spmatrix(dataset['features']).values, dtype = torch.float)
    y = torch.tensor(dataset['label'][0], dtype=torch.long)
    train_mask, val_mask, test_mask = gen_masks(y, 0.5, 0.3, 20)
    data = Data(x=x, y=y, edge_index=edge_index, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return data, num_features, num_classes

def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name)
    elif name.lower() in ['coauthorcs', 'coauthorphysics']:
        return get_coauthor(root, name[8:])
    elif name.lower() in ['amazoncomputers', 'amazonphoto']:
        return get_amazon(root, name[6:])
    elif name.lower() == 'wikics':
        return get_wikics(root)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    elif name.lower() == 'reddit2':
        return get_reddit(root)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() == 'yelp':
        return get_yelp(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root)
    elif name.lower() in ['yelpchi']:
        return get_yelpchi('./datasets/YelpChi.mat', 'yelpchi')
    else:
        raise NotImplementedError


def to_inductive(data):
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    i = 1
    while hasattr(data, f'x{i}'):
        data[f'x{i}'] = data[f'x{i}'][mask]
        i += 1
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def preprocess_data(model_config, data):
    loop, normalize = model_config['loop'], model_config['normalize']
    if loop:
        t = time.perf_counter()
        logger.info('Adding self-loops... ')
        data.adj_t = data.adj_t.set_diag()
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    if normalize:
        t = time.perf_counter()
        data.adj_t = gcn_norm(data.adj_t)
        logger.info(f'Done! [{time.perf_counter() - t:.2f}s]')


def prepare_dataset(model_config, data, remove_edge_index=True):
    train_data = to_inductive(data)
    train_data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(train_data.to('cuda'))
    data = T.ToSparseTensor(remove_edge_index=remove_edge_index)(data.to('cuda'))
    preprocess_data(model_config, train_data)
    preprocess_data(model_config, data)
    return train_data, data
