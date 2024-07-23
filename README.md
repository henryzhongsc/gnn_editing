# GNNs Also Deserve Editing, and They Need It More Than Once (SEED-GNN)

> This is the official implementation for our ICML 2024 SEED-GNN paper ([OpenReview](https://openreview.net/forum?id=rIc9adYbH2)).


Should you need to cite this work or find our codebase useful, here's the BibTex:

```
@inproceedings{zhong_le2024gnns_editing,
  title={{GNN}s Also Deserve Editing, and They Need It More Than Once},
  author={Shaochen Zhong and Duy Le and Zirui Liu and Zhimeng Jiang and Andrew Ye and Jiamu Zhang and Jiayi Yuan and Kaixiong Zhou and Zhaozhuo Xu and Jing Ma and Shuai Xu and Vipin Chaudhary and Xia Hu},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=rIc9adYbH2}
}
```

---
## Project Overview

`TL;DR` Model editing is popular in almost every other domain except graph because GNN editing is innately hard with no usable
method available. We reveal the roots of such complications and present the 1st GNN-editing work that lives up to real-life scrutiny.

`Abstract` Suppose a self-driving car is crashing into pedestrians, or a chatbot is instructing its users to conduct criminal wrongdoing; the stakeholders of such products will undoubtedly want to patch these catastrophic errors as soon as possible. To address such concerns, *Model Editing:* the study of efficiently patching model behaviors without significantly altering their general performance, has seen considerable activity, with hundreds of editing techniques developed in various domains such as CV and NLP. However, **the graph learning community has objectively fallen behind with only a few Graph Neural Network-compatible — and just one GNN-specific — model editing methods available**, where all of which are limited in their practical scope. We argue that the impracticality of these methods lies in their lack of *Sequential Editing Robustness:* the ability to edit multiple errors sequentially, and therefore fall short in effectiveness, as this approach mirrors how errors are discovered and addressed in the real world. In this paper, we delve into the specific reasons behind the difficulty of editing GNNs in succession and observe the root cause to be model overfitting. We subsequently propose a simple yet effective solution — SEED-GNN — by leveraging overfit-prevention techniques in a GNN-specific context to derive the first and only GNN model editing method that scales practically. Additionally, we formally frame the task paradigm of GNN editing and hope to inspire future research in this crucial but currently overlooked field. Please refer to our GitHub repository for code and checkpoints.


---
## Environment Setup

We provide you the minimum environment requirements to support the running of our project. This means there can be a slight difference depending on the actual automatic dependency solving result of your system.

```
pip install torch==2.0.0
pip install torch-scatter==2.1.1 torch-cluster==1.6.1 torch-spline-conv==1.2.2 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.0.0+${CUDA_VERSION}.html
# In our case CUDA_VERSION=cu117

pip install -r requirements.txt
```



---
## Experiment Reproduction

Before editing, it is a prerequsite to obtain the checkpoints for unedited models as baselines. We provide the vanilla training scripts for such baseline models as `scripts/pretrain/seed_gnn`, where one can run

```
bash cora.sh <output_dir_root> <dataset_dir>
```

to train the GCN, GrageSAGE, GIN, and GAT baseline models on the Cora dataset.


For editing, we provide a set of scripts at `scripts/edit`; where a script for editing a Cora-trained GCN model with SEED-GNN can be executed as the following:

```
bash scripts/edit/seed_gnn/gcn/cora.sh <output_folder_dir> <dataset_dir>
```

For readers interested in further developing our codebase, we provide a closer look at the `scripts/edit/seed_gnn/gcn/cora.sh` script so that you can modify it to your preference.

```
model="gcn" #options: gcn, gat, gin, sage
method="seed_gnn" #options: seed_gnn, egnn
dataset="cora" #options: amazoncomputers, amazonphoto, arxiv, coauthorcs, cora, products
output_dir_root=$1
dataset_dir=$2


python main.py \
        --exp_desc="edit_${model}_${method}" \
        --pipeline_config_dir="config/pipeline_config/${method}/${model}/${dataset}.json" \
        --eval_config_dir="config/eval_config/edit_gnn/${dataset}.json" \
        --task="edit" \
        --output_folder_dir="${output_dir_root}/results/${method}/${model}/${dataset}/" \
        --pretrain_output_dir="${output_dir_root}/edit_ckpts" \
        --dataset_dir="${dataset_dir}"

```

The above script has the argparse definitions of:

* `exp_desc`: a descriptive value that will show on the output of the experiment to help with bookkeeping. It won't affect the running of the program.
* `pipeline_config_dir`: directory of a pipeline config file (usually from `config/pipeline_config`) that defines some method-related settings and hyperparameters.
* `eval_config_dir`: directory of an eval config file (usually from `config/eval_config`) that defines some dataset-related settings and hyperparameters.
* `task`: set to `edit` since we are doing model editing.
* `output_folder_dir`: output directory of results.
* `pretrain_output_dir`: directory of unedited model checkpoints.
* `dataset_dir`: directory datasets.



---

## Results Digestion

Once the experiment is executed, you should be able to monitor real time printouts in the terminal as well as the `exp.log` file in the `<output_folder_dir>` folder you supplied above. Once the experiment is successfully concluded, you shall find the following items in the `<output_folder_dir>` folder:


* `input_config` folder: This folder contains an `input_pipeline_config.json` and an`input_eval_config.json`. These are the carbon copy of the configs supplied to the `pipeline_config_dir` and `eval_config_dir` arguments of the editing script. Such configs are copied here for easy replication purposes as these two configs basically define an experiment.
* `output_config.json`: This file provides a fuse of the above two input configs and some management information (e.g., start/end time of a job). Most importantly, it highlights the main reported metrics under the key `eval_results,` which are as follows:
    * `bef_edit_tst_acc`: Pre-edit accuracy of a model. **PE Acc.** in the main tables of our paper.
    * `selected_result`: A dictionary in the format of `{nth_edit: [test_drawdown, success_rate]...}`. In this case, we highlight the **1, 10, 25, and 50th edit as Test Drawdown (Success Rate)** in the main tables of our paper.
    * `highest_dd`: Highest Test Drawdown happened during all edits. **Max DD** in the main tables of our paper.
    * `average_dd`: Mean Test Drawdown of all edits. **Avg DD** in the main tables of our paper.
    * `average_success_rate`: Mean Success Rate of all edits. **Avg SR** in the main tables of our paper.
    * If you are only interested in comparing metrics reported in the main tables (like Tables 5 and 6) of our paper, this is the only place you need to look at. We put such results alongside the two input configs so there is no chance of attributing the results to a wrong setting.
* `raw_results.json`: This file registers the fine grain results of the concluded experiment, even if they are not reported in our paper — e.g., the Test Drawdown and Success Rate after *every* edit, standard deviation, etc. These results help for up-close diagnostics of a particular run.
* `exp.log`: This is a carbon copy of the real-time printouts to the terminal.
