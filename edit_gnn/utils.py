import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch_geometric.data.data import Data

from models.base import BaseModel

def grab_input(data: Data, indices=None):
    x = data.x
    i = 1
    xs = [x]
    # for SIGN
    while hasattr(data, f'x{i}'):
        xs.append(getattr(data, f'x{i}'))
        i += 1
    return {"x": data.x, 'adj_t': data.adj_t}


@torch.no_grad()
def prediction(model: BaseModel, data: Data):
    model.eval()
    input = grab_input(data)
    return model(**input)


def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]
    if y.dim() == 1:
        try:
            return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        except ZeroDivisionError:
            return 0.

    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


@torch.no_grad()
def test(model: BaseModel, data: Data, specific_class: int = None):
    model.eval()
    out = prediction(model, data)
    y_true = data.y
    train_mask = data.train_mask
    valid_mask = data.val_mask
    test_mask = data.test_mask
    if specific_class is not None:
        mask = data.y == specific_class
        out = out[mask]
        y_true = y_true[mask]
        train_mask = train_mask[mask]
        valid_mask = valid_mask[mask]
        test_mask = test_mask[mask]
    train_acc = compute_micro_f1(out, y_true, train_mask)
    valid_acc = compute_micro_f1(out, y_true, valid_mask)
    test_acc = compute_micro_f1(out, y_true, test_mask)
    return train_acc, valid_acc, test_acc


def select_edit_target_nodes(model: BaseModel,
                            whole_data: Data,
                            num_classes: int,
                            num_samples: int,
                            from_valid_set: bool = True):
        model.eval()
        bef_edit_logits = prediction(model, whole_data)
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        val_y_true = whole_data.y[whole_data.val_mask]
        val_y_pred = bef_edit_pred[whole_data.val_mask]
        if from_valid_set:
            nodes_set = whole_data.val_mask.nonzero().squeeze()
        else:
            nodes_set = whole_data.train_mask.nonzero().squeeze()

        wrong_pred_set = val_y_pred.ne(val_y_true).nonzero()
        val_node_idx_2flip = wrong_pred_set[torch.randperm(len(wrong_pred_set))[:num_samples]]
        node_idx_2flip = nodes_set[val_node_idx_2flip]
        flipped_label = whole_data.y[node_idx_2flip]

        return node_idx_2flip, flipped_label


def process_edit_results(bef_edit_results, raw_results):
    bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
    success_rate = 0
    success_list = []
    average_dd = []
    highest_dd = []
    lowest_dd = []
    test_dd_std = 0
    selected_result = []

    train_acc, val_acc, test_acc, succeses, steps, mem, tot_time = zip(*raw_results)
    tra_drawdown = bef_edit_tra_acc - train_acc[-1]
    val_drawdown = bef_edit_val_acc - val_acc[-1]
    test_drawdown = test_drawdown = np.round((np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals = 3).tolist()
    average_dd = np.round(np.mean(np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals=3) * 100

    success_rate = np.round(np.mean(succeses), decimals = 3).tolist()
    success_list = np.round(np.array(succeses), decimals = 3).tolist()

    test_drawdown = [test_drawdown * 100] if not isinstance(test_drawdown, list) else [round(d * 100, 1) for d in test_drawdown]
    test_dd_std = np.std(test_drawdown)
    highest_dd = max(enumerate(test_drawdown), key=lambda x: x[1])
    lowest_dd = min(enumerate(test_drawdown), key=lambda x: x[1])
    selected_result = {
        '1': (test_drawdown[0], success_list[0]),
        '10': (test_drawdown[9], success_list[9]),
        '25': (test_drawdown[24], success_list[24]),
        '50': (test_drawdown[49], success_list[49])
    }
    mem_result = {}
    time_result = {}
    mem_result = {
        'max_memory': str(np.round(np.max(mem), decimals=3)) + 'MB'
    }
    time_result = {
        '1': str(np.round(tot_time[0], decimals=3)),
        '10': str(np.round(tot_time[9], decimals=3)),
        '25': str(np.round(tot_time[24], decimals=3)),
        '50': str(np.round(tot_time[49], decimals=3)),
        'total_time': np.sum(tot_time)
    }

    return dict(bef_edit_tra_acc=bef_edit_tra_acc,
                bef_edit_val_acc=bef_edit_val_acc,
                bef_edit_tst_acc=bef_edit_tst_acc,
                tra_drawdown=tra_drawdown * 100,
                val_drawdown=val_drawdown * 100,
                test_drawdown=test_drawdown,
                success_rate=success_rate,
                success_list = success_list,
                average_dd = average_dd,
                test_dd_std=test_dd_std,
                highest_dd = highest_dd,
                lowest_dd = lowest_dd,
                selected_result = selected_result,
                mean_complexity=np.mean(steps),
                memory_result=mem_result,
                time_result=time_result
                )


def success_rate(model, idx, label, whole_data):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    model.eval()

    input = grab_input(whole_data)
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        y_pred = out.argmax(dim=-1)[idx]
    success = int(y_pred.eq(label).sum()) / label.size(0)
    torch.cuda.synchronize()
    
    return success
