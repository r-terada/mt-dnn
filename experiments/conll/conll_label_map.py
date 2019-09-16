# Copyright (c) Microsoft. All rights reserved.

from data_utils.vocab import Vocabulary
from data_utils.task_def import TaskType, DataFormat
from data_utils.metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman, compute_precision, compute_recall

# conll2003 ner
NERLabelMapper = Vocabulary(True)
for l in ["X", "[CLS]", "[SEP]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]:
    NERLabelMapper.add(l)

GLOBAL_MAP = {
    'ner': NERLabelMapper,
}

# number of class
DATA_META = {
    'ner': len(NERLabelMapper.get_vocab_list()),
}

DATA_TYPE = {
    'ner': DataFormat.PremiseOnly,
}

TASK_TYPE = {
    'ner': TaskType.SequenceLabeling,
}

# see: metrics.py
METRIC_META = {
    'ner': [8, 9, 10, 11, 12, 13],
}

DECODER_OPT = {
    'ner': 2,
}
