# Copyright (c) Microsoft. All rights reserved.
import numpy as np
from sklearn.metrics import classification_report
from random import shuffle
from data_utils.metrics import Metric, METRIC_FUNC
from data_utils.task_def import DataFormat
from functools import reduce


def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))


def _flatten_list(l):
    return reduce(lambda a, b: a + b, l)


def eval_model(model, data, metric_meta, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        batch_size = len(gold)
        true_seq_length = [len(g) for g in gold]
        batch_seq_length = int(len(pred) / batch_size)
        preds = []
        for i, t_l in enumerate(true_seq_length):
            preds.append(pred[i * batch_seq_length:(i * batch_seq_length + t_l)])
        predictions.extend(preds)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])

    # remove ["O", "X", "[CLS]", "[SEP]"] from evaluation
    # all task must add labels in the order below
    # LabelMapper.add("X")
    # LabelMapper.add("[CLS]")
    # LabelMapper.add("[SEP]")
    # LabelMapper.add("O")
    use_indices = [label > 3 for label in _flatten_list(golds)]
    if with_label:
        if any(use_indices):
            print(classification_report(
                np.array(_flatten_list(golds))[use_indices],
                np.array(_flatten_list(predictions))[use_indices],
            ))
        for mm in metric_meta:
            metric_name = mm.name
            metric_func = METRIC_FUNC[mm]
            metric = metric_func(
                np.array(_flatten_list(predictions))[use_indices],
                np.array(_flatten_list(golds))[use_indices]
            )
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids

