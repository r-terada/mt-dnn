#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "run_ner.sh <batch_size> <gpu>"
  exit 1
fi
prefix="mt-dnn-nerall-pos"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

train_datasets="nerall,pos"
test_datasets="nerall"
MODEL_ROOT="checkpoints"

BERT_PATH="mt_dnn_models/Japanese_L-12_H-768_A-12_E-30_BPE/pytorch_model.bin"
BERT_CONFIG_PATH="mt_dnn_models/Japanese_L-12_H-768_A-12_E-30_BPE/bert_config.json"
DATA_DIR="data/bccwj_all_class"
TASK_DEF_PATH="experiments/japanese/japanese_task_def.yml"

optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="2e-5"

model_dir="checkpoints/${prefix}_${optim}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python train.py --data_dir ${DATA_DIR} --bert_config_path ${BERT_CONFIG_PATH} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --multi_gpu_on --task_def ${TASK_DEF_PATH}
