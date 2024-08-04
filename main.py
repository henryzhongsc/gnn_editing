import sys
import os
import json
import datetime
from zoneinfo import ZoneInfo

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(current_dir)
sys.path.append(base_dir)
os.chdir(base_dir)

import main_utils as main_utils
from pipelines.seed_gnn.eval_edit_gnn import eval_edit_gnn
from pipelines.seed_gnn.pretrain_gnn import pretrain_gnn
from constants import NODE_CLASSIFICATION_DATASETS, SEED

main_utils.set_seeds_all(SEED)

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = main_utils.parse_args()
config = main_utils.register_args_and_configs(args)
logger = main_utils.set_logger(args.output_folder_dir, args)


logger.info(f"Experiment {config['management']['exp_desc']} (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))

if config['eval_params']['dataset'] in NODE_CLASSIFICATION_DATASETS:
    if config['management']['task'] == 'edit':
        raw_results, processed_results = eval_edit_gnn(config)
        main_utils.register_result(raw_results, config)
        config['eval_results'] = processed_results
    elif config['management']['task'] == 'pretrain':
        pretrain_gnn(config)
    else:
        logger.error(f"Invalid args.task input: {config['management']['task']}.")
        raise ValueError
else:
    logger.error(f"Invalid config['eval_params']['dataset'] input: {config['eval_params']['dataset']}.")
    raise ValueError

end_time = datetime.datetime.now(ct_timezone)
main_utils.register_exp_time(start_time, end_time, config)
main_utils.register_output_config(config)
logger.info(f"Experiment {config['management']['exp_desc']} ended at {end_time}. Duration: {config['management']['exp_duration']}")
