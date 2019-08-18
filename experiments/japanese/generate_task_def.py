import os
from sys import path
path.append(os.getcwd())
from data_utils.metrics import Metric
from data_utils.task_def import DataFormat, TaskType
from experiments.japanese.bccwj_label_map import GLOBAL_MAP, METRIC_META, SAN_META

task_def_dic = {}
dropout_p_map = {
    "ner": 0.1,
    "ner_ene": 0.1,
    "pos": 0.1
}
for task in ['ner', 'ner_ene', 'pos']:
    task_type = TaskType.SequenceLabeling
    data_format = DataFormat.PremiseOnly

    labels = None
    labels = GLOBAL_MAP[task].get_vocab_list()

    n_class = len(labels)
    metric_meta = tuple(Metric(metric_no).name for metric_no in METRIC_META[task])
    enable_san = SAN_META[task]

    task_def = {"task_type": task_type.name,
                "data_format": data_format.name,
                "n_class": n_class,
                "metric_meta": metric_meta,
                "enable_san": enable_san
                }

    if labels is not None:
        task_def["labels"] = labels

    dropout_p = dropout_p_map.get(task, None)
    if dropout_p is not None:
        task_def["dropout_p"] = dropout_p

    task_def_dic[task] = task_def

import yaml

with open("experiments/japanese/japanese_task_def.yml", "w", encoding="utf-8") as f:
    yaml.safe_dump(task_def_dic, f)
