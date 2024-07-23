from tqdm import tqdm
from copy import deepcopy

from edit_gnn.utils import test, success_rate
from seed_gnn.utils import get_optimizer, edit

def egnn_edit(config, model, node_idx_2flip, flipped_label, whole_data, max_num_step):
        model.train()
        model = deepcopy(model)
        optimizer = opt = get_optimizer(config['pipeline_params'], model)
        raw_results = []
        i = 0

        for idx, f_label in tqdm(zip(node_idx_2flip, flipped_label)):
            i = i + 1
            edited_model, success, loss, steps, mem, tot_time = edit(model, whole_data, idx, f_label, optimizer, max_num_step)
            success = success_rate(model, node_idx_2flip[:i].squeeze(dim=1), flipped_label[:i].squeeze(dim=1), whole_data)
            res = [*test(edited_model, whole_data), success, steps]
            res.append(mem)
            res.append(tot_time)
            raw_results.append(res)
        
        return raw_results
