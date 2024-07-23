method="seed_gnn"
dataset="arxiv"
output_dir_root=$1
dataset_dir=$2


for model in gcn sage gin gat; do
        python main.py \
                --exp_desc="pretrain_${model}_${method}" \
                --pipeline_config_dir="config/pipeline_config/${method}/${model}/${dataset}.json" \
                --eval_config_dir="config/eval_config/edit_gnn/${dataset}.json" \
                --task="pretrain" \
                --output_folder_dir="${output_dir_root}/results/${method}/${model}/${dataset}/" \
                --pretrain_output_dir="${output_dir_root}/edit_ckpts"\
                --dataset_dir="${dataset_dir}"
done