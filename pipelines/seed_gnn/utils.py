import os
import time
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
logger = logging.getLogger("main")

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data

from models.base import BaseModel
from edit_gnn.utils import grab_input, test

def get_optimizer(model_config, model, pretrain=False):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=model_config['pretrain_lr'] if pretrain else model_config['edit_lr'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=model_config['pretrain_lr'] if pretrain else model_config['edit_lr'])
    else:
        raise NotImplementedError

    return optimizer
    
def sorted_checkpoints(
        checkpoint_prefix, best_model_checkpoint, output_dir=None
    ) -> List[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted


def save_model(model, save_path, checkpoint_prefix: str, epoch: int):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_model_checkpoint = os.path.join(save_path, f'{checkpoint_prefix}_{epoch}.pt')
    torch.save(model.state_dict(), best_model_checkpoint)
    checkpoints_sorted = sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, save_path)
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        os.remove(checkpoint)


def finetune_mlp(config, model, whole_data, train_data, batch_size, iters):
        input = grab_input(train_data)
        model.eval()

        # get the original GNN output embedding
        model.mlp_freezed = True
        with torch.no_grad():
            gnn_output = model(**input)
            model.gnn_output = model(**grab_input(whole_data)).cpu()
            log_gnn_output = F.log_softmax(gnn_output, dim=-1)
    
        # here we enable the MLP to be trained
        model.freeze_module(train=False)
        opt = get_optimizer(config['pipeline_params'], model)
        logger.info('start finetuning MLP')
        s = time.time()
        torch.cuda.synchronize()
        for i in tqdm(range(iters)):
            opt.zero_grad()
            idx = np.random.choice(train_data.num_nodes, batch_size)
            idx = torch.from_numpy(idx).to(gnn_output.device)
            MLP_output = model.MLP(train_data.x[idx])
            cur_batch_gnn_output = gnn_output[idx]
            log_prob = F.log_softmax(MLP_output + cur_batch_gnn_output, dim=-1)
            main_loss = F.cross_entropy(MLP_output + gnn_output[idx], train_data.y[idx])
            kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
            (kl_loss + main_loss).backward()
            opt.step()
        torch.cuda.synchronize()
        e = time.time()
        logger.info(f'fine tune MLP used: {e - s} sec.')


def finetune_gnn_mlp(config, model, whole_data, train_data):
    model.freeze_module(train=False)
    dataset = config['eval_params']['dataset']
    if dataset == 'flickr' or (dataset == 'reddit2' and config['pipeline_params']['model_name']) or \
        (dataset in ['amazoncomputers', 'amazonphoto', 'coauthorcs', 'coauthorphysics']):
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=512, iters=100)
    else:
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=32, iters=100)
    bef_edit_ft_results = test(model, whole_data)
    ft_train_acc, ft_valid_acc, ft_test_acc = bef_edit_ft_results
    logger.info(f'before edit, after fine tune, train acc {ft_train_acc}, valid acc {ft_valid_acc}, test acc {ft_test_acc}')

    return bef_edit_ft_results


def bef_edit_check(model, whole_data, idx, label, curr_edit_target):
    model.eval()
    torch.cuda.synchronize()
    input = grab_input(whole_data)
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        y_pred = out.argmax(dim=-1)[idx]

    if label.shape[0] == 1:
        success = (y_pred == label)
    else:
        success = 1.0 if y_pred.eq(label)[curr_edit_target] else 0.0
    torch.cuda.synchronize()

    return success


def single_edit(model, whole_data, idx, label, optimizer, max_num_step, num_edit_targets=1):
    s = time.time()
    loss_op = F.cross_entropy
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    success = False

    for step in range(1, max_num_step + 1):
        optimizer.zero_grad()
        input = grab_input(whole_data)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
            out = model.fast_forward(input['x'][idx], idx)
            loss = loss_op(out, label)
            y_pred = out.argmax(dim=-1)
        else:
            out = model(**input)
            loss = loss_op(out[idx], label)
            y_pred = out.argmax(dim=-1)[idx]
        loss.backward()
        optimizer.step()
        if label.shape[0] == 1:
            success = y_pred == label
        else:
            success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets])[:num_edit_targets].sum()) / num_edit_targets
        if success == 1.:
            break
    
    torch.cuda.synchronize()
    e = time.time()
    logger.info(f'max allocated mem: {torch.cuda.max_memory_allocated() / (1024**2)} MB')
    logger.info(f'edit time: {e - s}')
    return model, success, loss, step, torch.cuda.max_memory_allocated() / (1024**2), e - s



def edit(model, whole_data, idx, f_label, optimizer, max_num_step,
        num_edit_targets=1, curr_edit_target=0):
    bef_edit_success = bef_edit_check(model, whole_data, idx, f_label,curr_edit_target=curr_edit_target)
    if bef_edit_success == 1.:
        return model, bef_edit_success, 0, 0, 0, 0

    return single_edit(model, whole_data, idx, f_label, optimizer, max_num_step, num_edit_targets=num_edit_targets)

def process_raw_exp_results(raw_results):
    processed_results = {}

    processed_results['bef_edit_tst_acc'] = raw_results['bef_edit_tst_acc']
    processed_results['selected_result'] = raw_results['selected_result']
    processed_results['highest_dd'] = raw_results['highest_dd']
    processed_results['average_dd'] = raw_results['average_dd']
    processed_results['average_sucess_rate'] = raw_results['average_success_rate']
    
    return processed_results
