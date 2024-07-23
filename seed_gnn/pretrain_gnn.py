import os
import logging
from copy import deepcopy
from pathlib import Path
logger = logging.getLogger("main")

import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data

import models as models
from constants import SEED
from main_utils import set_seeds_all
from models.base import BaseModel
from data import get_data, prepare_dataset
from edit_gnn.utils import grab_input, test
from seed_gnn.utils import get_optimizer, save_model
from seed_gnn.seed_gnn_logging import Logger


def train_loop(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_data: Data,
    loss_op):
    model.train()
    optimizer.zero_grad()
    input = grab_input(train_data)
    out = model(**input)
    loss = loss_op(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def train(config):
    set_seeds_all(SEED)
    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    data, num_features, num_classes = get_data(config['management']['dataset_dir'], config['eval_params']['dataset'])
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])

    model = MODEL_FAMILY(
        in_channels=num_features, 
        out_channels=num_classes, 
        load_pretrained_backbone = config['pipeline_params']['load_pretrained_backbone'], 
        saved_ckpt_path = save_path, 
        **config['pipeline_params']['architecture']
    )
    model.cuda()
    logger.info(model)

    train_data, whole_data = prepare_dataset(config['pipeline_params'], data, remove_edge_index=True)
    del data
    logger.info(f'training data: {train_data}')
    logger.info(f'whole data: {whole_data}')

    if not config['pipeline_params']['load_pretrained_backbone']:
        model.reset_parameters()
    optimizer = get_optimizer(config['pipeline_params'], model, pretrain=True)
    best_val = -1.
    loss_op = F.cross_entropy
    train_logger = Logger()
    checkpoint_prefix = f"{config['pipeline_params']['model_name']}"
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])

    for epoch in range(1, config['pipeline_params']['epochs'] + 1):
        if not config['pipeline_params']['load_pretrained_backbone']:
            train_loss = train_loop(model, optimizer, train_data, loss_op)
        result = test(model, whole_data)
        train_logger.add_result(result)
        train_acc, valid_acc, test_acc = result
        # save the model with the best valid acc
        if valid_acc > best_val:
            save_model(model, save_path, checkpoint_prefix, epoch)
            best_val = valid_acc

        logger.info(f'Epoch: {epoch:02d}, '
                    f'Train f1: {100 * train_acc:.2f}%, '
                    f'Valid f1: {100 * valid_acc:.2f}% '
                    f'Test f1: {100 * test_acc:.2f}%')

    train_logger.print_statistics()


def pretrain_gnn(config):
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])
    checkpoints = [str(x) for x in Path(save_path).glob(f"{config['pipeline_params']['model_name'].replace('_MLP', '')}_*.pt")]
    
    # For GNN + MLP, we need to pretrain the GNN backbone first, then the GNN + MLP
    if '_MLP' in config['pipeline_params']['model_name'] and len(checkpoints) < 1:
        backbone_config = deepcopy(config)
        backbone_config['pipeline_params']['model_name'] = backbone_config['pipeline_params']['model_name'].replace('_MLP', '')
        backbone_config['pipeline_params']['load_pretrained_backbone'] = False
        train(backbone_config)
    
    train(config)
