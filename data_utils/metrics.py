# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr

def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

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

def compute_recall(predicts, labels):
    return 100.0 * recall_score(labels, predicts)


class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    Precision = 5
    Recall = 6


METRIC_FUNC = {
 Metric.ACC: compute_acc,
 Metric.F1: compute_f1,
 Metric.MCC: compute_mcc,
 Metric.Pearson: compute_pearson,
 Metric.Spearman: compute_spearman,
 Metric.Precision: compute_precision,
 Metric.Recall: compute_recall,
}