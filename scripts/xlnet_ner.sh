#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
prefix="xlnet-ner"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="ner"
test_datasets="ner"
MODEL_ROOT="checkpoints"
DATA_DIR="../data/canonical_data/xlnet_cased"

answer_opt=2
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
task_def="../experiments/conll/conll_task_def_xlnet.yml"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python ../train.py --data_dir ${DATA_DIR} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --task_def ${task_def}
