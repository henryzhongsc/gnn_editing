import os
import logging
logger = logging.getLogger("main")

import torch

import models as models
from main_utils import set_seeds_all
from data import get_data, prepare_dataset
from constants import SEED
from edit_gnn.utils import test, select_edit_target_nodes, process_edit_results
from pipelines.seed_gnn.utils import finetune_gnn_mlp, process_raw_exp_results
from pipelines.seed_gnn.egnn import egnn_edit
from pipelines.seed_gnn.seed_gnn import seed_gnn_edit

def eval_edit_gnn(config):
    set_seeds_all(SEED)
    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    data, num_features, num_classes = get_data(config['management']['dataset_dir'], config['eval_params']['dataset'])
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])
    
    model = MODEL_FAMILY.from_pretrained(
                                in_channels=num_features,
                                out_channels=num_classes,
                                saved_ckpt_path=save_path,
                                **config['pipeline_params']['architecture']
                                )
    logger.info(model)
    model.cuda()

    train_data, whole_data = prepare_dataset(config['pipeline_params'], data, remove_edge_index=False)
    del data
    logger.info(f'training data: {train_data}')
    logger.info(f'whole data: {whole_data}')

    bef_edit_results = test(model, whole_data)
    bef_edit_train_acc, bef_edit_valid_acc, bef_edit_test_acc = bef_edit_results
    logger.info(f'before edit, train acc {bef_edit_train_acc}, valid acc {bef_edit_valid_acc}, test acc {bef_edit_test_acc}')

    node_idx_2flip, flipped_label = select_edit_target_nodes(model=model,
                                                            whole_data=whole_data,
                                                            num_classes=num_classes,
                                                            num_samples=config['eval_params']['num_targets'],
                                                            from_valid_set=True)
    node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()

    if '_MLP' in config['pipeline_params']['model_name']:
        bef_edit_ft_results = finetune_gnn_mlp(config, model, whole_data, train_data)

    raw_results = None
    if config['pipeline_params']['method'] == 'egnn':
        raw_results = egnn_edit(config=config,
                model=model,
                node_idx_2flip=node_idx_2flip,
                flipped_label=flipped_label,
                whole_data=whole_data,
                max_num_step=config['pipeline_params']['max_num_edit_steps'])
    elif config['pipeline_params']['method'] == 'seed_gnn':
        raw_results = seed_gnn_edit(
            config=config,
            model=model,
            node_idx_2flip=node_idx_2flip,
            flipped_label=flipped_label,
            whole_data=whole_data,
            max_num_step=config['pipeline_params']['max_num_edit_steps'])
    else:
        logger.info(f"Editing method {config['pipeline_params']['method']} is not implemented.")
        raise NotImplementedError

    raw_results = process_edit_results(bef_edit_results, raw_results)
    processed_results = process_raw_exp_results(raw_results)

    return raw_results, processed_results
