# Copyright (c) Microsoft. All rights reserved.

from data_utils.vocab import Vocabulary
from data_utils.task_def import TaskType, DataFormat
from data_utils.metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman, compute_precision, compute_recall

# conll2003 ner
NERLabelMapper = Vocabulary(True)
for l in ["X", "[CLS]", "[SEP]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]:
    NERLabelMapper.add(l)

ChunkLabelMapper = Vocabulary(True)
for l in ["X", "[CLS]", "[SEP]", "O", 'I-ADJP', 'B-ADVP', 'B-ADJP', 'I-PRT', 'O', 'B-NP', 'B-SBAR', 'B-CONJP', 'I-CONJP', 'I-LST', 'B-INTJ', 'I-PP', 'B-LST', 'I-VP', 'B-PP', 'B-VP', 'B-PRT', 'I-NP', 'I-ADVP', 'I-SBAR', 'I-INTJ']:
    ChunkLabelMapper.add(l)

POSLabelMapper = Vocabulary(True)
for l in ["X", "[CLS]", "[SEP]", 'POS', 'NN|SYM', 'RB', 'TO', 'LS', 'SYM', '.', 'WP', 'EX', 'JJR', 'VBD', "''", 'VBG', 'PRP', 'WRB', 'CD', '(', ':', 'FW', 'UH', 'NNP', 'RP', 'MD', 'NN', 'NNS', 'PDT', ',', ')', 'WDT', 'NNPS', 'VBP', 'RBR', 'VBZ', 'RBS', 'VB', 'CC', 'PRP$', 'JJS', 'VBN', 'IN', 'DT', '"', 'JJ', 'WP$', '$']:
    POSLabelMapper.add(l)

GLOBAL_MAP = {
    'ner': NERLabelMapper,
    'chunking': ChunkLabelMapper,
    'pos': POSLabelMapper,
}

# number of class
DATA_META = {
    'ner': len(NERLabelMapper.get_vocab_list()),
    'chunking': len(ChunkLabelMapper.get_vocab_list()),
    'pos': len(POSLabelMapper.get_vocab_list()),
}

DATA_TYPE = {
    'ner': DataFormat.PremiseOnly,
    'chunking': DataFormat.PremiseOnly,
    'pos': DataFormat.PremiseOnly,
}

TASK_TYPE = {
    'ner': TaskType.SequenceLabeling,
    'chunking': TaskType.SequenceLabeling,
    'pos': TaskType.SequenceLabeling,
}

# see: metrics.py
METRIC_META = {
    'ner': [8, 9, 10, 11, 12, 13],
    'chunking': [8, 9, 10, 11, 12, 13],
    'pos': [8, 9, 10, 11, 12, 13],
}

DECODER_OPT = {
    'ner': 2,
    'chunking': 2,
    'pos': 2,
}
