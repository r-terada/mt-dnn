# Copyright (c) Microsoft. All rights reserved.
from enum import Enum

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
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

def compute_auc(predicts, labels):
    auc = roc_auc_score(labels, predicts)
    return 100.0 * auc

def compute_micro_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='micro')

def compute_macro_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts, average='macro')

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
    AUC = 5
    Precision = 6
    Recall = 7
    MICRO_F1 = 8
    MACRO_F1 = 9
    MICRO_Precision = 10
    MACRO_Precision = 11
    MICRO_Recall = 12
    MACRO_Recall = 13


METRIC_FUNC = {
    Metric.ACC: compute_acc,
    Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.Precision: compute_precision,
    Metric.Recall: compute_recall,
    Metric.MICRO_F1: compute_micro_f1,
    Metric.MACRO_F1: compute_macro_f1,
    Metric.MICRO_Precision: compute_micro_precision,
    Metric.MACRO_Precision: compute_macro_precision,
    Metric.MICRO_Recall: compute_micro_recall,
    Metric.MACRO_Recall: compute_macro_recall,
}

def calc_metrics(metric_meta, golds, predictions, scores):
    metrics = {}
    for mm in metric_meta:
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.F1, Metric.MCC):
            metric = metric_func(predictions, golds)
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(golds), "AUC is only valid for binary classification problem"
                scores = scores[1::2]
            metric = metric_func(scores, golds)
        metrics[metric_name] = metric
    return metrics
