import os
from sys import path
path.append(os.getcwd())

from data_utils.metrics import Metric
from data_utils.task_def import TaskType, DataFormat, EncoderModelType
from experiments.conll.conll_label_map import DATA_TYPE, GLOBAL_MAP, TASK_TYPE, DATA_META, METRIC_META, DECODER_OPT

task_def_dic = {}
task_def_dic_xlnet = {}
task_def_dic_roberta = {}
dropout_p_map = {}
for task in TASK_TYPE.keys():
    task_type = TASK_TYPE[task]
    data_format = DATA_TYPE[task]

    labels = None
    labels = GLOBAL_MAP[task].get_vocab_list()

    n_class = DATA_META[task]
    metric_meta = tuple(Metric(metric_no).name for metric_no in METRIC_META[task])
    decoder_opt = DECODER_OPT[task]

    task_def = {
        "task_type": task_type.name,
        "data_format": data_format.name,
        "n_class": n_class,
        "metric_meta": metric_meta,
        "decoder_opt": decoder_opt,
        "labels": labels
    }

    task_def_dic[task] = task_def

    task_def_xlnet = task_def.copy()
    task_def_xlnet['encoder_type'] = EncoderModelType.XLNET.name
    task_def_dic_xlnet[task] = task_def_xlnet

    task_def_roberta = task_def.copy()
    task_def_roberta['encoder_type'] = EncoderModelType.ROBERTA.name
    task_def_dic_roberta[task] = task_def_roberta

import yaml

yaml.safe_dump(task_def_dic, open("experiments/conll/conll_task_def.yml", "w"))
yaml.safe_dump(task_def_dic_xlnet, open("experiments/conll/conll_task_def_xlnet.yml", "w"))
yaml.safe_dump(task_def_dic_roberta, open("experiments/conll/conll_task_def_roberta.yml", "w"))
