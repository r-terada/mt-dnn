# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

def compute_micro_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='micro')

def compute_macro_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='macro')

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_precision(predicts, labels):
    return 100.0 * precision_score(labels, predicts)

def compute_micro_precision(predicts, labels):
    return 100.0 * precision_score(labels, predicts, average='micro')

def compute_macro_precision(predicts, labels):
    return 100.0 * precision_score(labels, predicts, average='macro')

def compute_recall(predicts, labels):
    return 100.0 * recall_score(labels, predicts)

def compute_micro_recall(predicts, labels):
    return 100.0 * recall_score(labels, predicts, average='micro')

def compute_macro_recall(predicts, labels):
    return 100.0 * recall_score(labels, predicts, average='macro')

class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    Precision = 5
    Recall = 6
    MICRO_F1 = 7
    MACRO_F1 = 8
    MICRO_Precision = 9
    MACRO_Precision = 10
    MICRO_Recall = 11
    MACRO_Recall = 12


METRIC_FUNC = {
 Metric.ACC: compute_acc,
 Metric.F1: compute_f1,
 Metric.MCC: compute_mcc,
 Metric.Pearson: compute_pearson,
 Metric.Spearman: compute_spearman,
 Metric.Precision: compute_precision,
 Metric.Recall: compute_recall,
 Metric.MICRO_F1: compute_micro_f1,
 Metric.MACRO_F1: compute_macro_f1,
 Metric.MICRO_Precision: compute_micro_precision,
 Metric.MACRO_Precision: compute_macro_precision,
 Metric.MICRO_Recall: compute_micro_recall,
 Metric.MACRO_Recall: compute_macro_recall,
}