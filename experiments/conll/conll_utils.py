import numpy as np
from random import shuffle
from functools import reduce, partial
from data_utils.task_def import DataFormat
from data_utils.metrics import METRIC_FUNC


def load_conll(file, label_field):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            if contents.startswith("-DOCSTART-") or len(contents) == 0:
                    continue
            
            if len(words) > 0 and words[-1] == '.':
                w = ' '.join([word for word in words if len(word) > 0])
                l = ' '.join([label for label in labels if len(label) > 0])
                sample = {'uid': str(cnt), 'premise': w, 'label': l}
                rows.append(sample)
                cnt += 1
                words = []
                labels = []
            else:
                word = contents.split(' ')[0]
                label = contents.split(' ')[label_field]
                words.append(word)
                labels.append(label)

    return rows


load_ner = partial(load_conll, label_field=3)
load_chunking = partial(load_conll, label_field=2)
load_pos = partial(load_conll, label_field=1)


def dump_rows(rows, out_path):
    """
    output files should have following format
    :param rows:
    :param out_path:
    :return:
    """

    with open(out_path, "w", encoding="utf-8") as out_f:
        for row in rows:
            for col in ["uid", "label", "premise"]:
                if "\t" in str(row[col]):
                    import pdb; pdb.set_trace()
            out_f.write("%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"]))


def _flatten_list(l):
    return reduce(lambda a, b: a + b, l)


def eval_model(model, data, metric_meta, vocab, use_cuda=True, with_label=True, beam_search=True, beam_width=5, export_file=None):
    label2id = vocab.tok2ind
    id2label = vocab.ind2tok
    n_labels = len(label2id)
    all_labels = [id2label[i] for i in range(n_labels)]

    data.reset()
    if use_cuda:
        model.cuda()
    inputs = []
    predictions = []
    golds = []
    scores = []
    ids = []
    for batch_meta, batch_data in data:
        # batch_data: input, _, mask
        input_data = batch_data[0]  # (batch_size, batch_seq_length)
        score, pred, gold = model.predict(batch_meta, batch_data)
        batch_size = len(gold)
        true_seq_length = [len(g) for g in gold]
        batch_seq_length = int(len(pred) / batch_size)

        # Make (inputs_, preds, gold)'s shape as (batch_size, true_seq_length_i)
        inputs_ = [input_sentence[:t_l].cpu().detach().numpy().tolist() for input_sentence, t_l in zip(input_data, true_seq_length)]
        inputs.extend(inputs_)

        preds = [pred[i * batch_seq_length:(i * batch_seq_length + t_l)] for i, t_l in enumerate(true_seq_length)]
        predictions.extend(preds)

        score_ = [[score[(i * batch_seq_length + j) * n_labels:(i * batch_seq_length + j + 1) * n_labels] for j in range(t_l)] for i, t_l in enumerate(true_seq_length)]
        scores.extend(score_)

        golds.extend(gold)
        ids.extend(batch_meta['uids'])

    # remove ["O", "X", "[CLS]", "[SEP]"] from evaluation
    # all task must add labels in the order below
    # LabelMapper.add("X")
    # LabelMapper.add("[CLS]")
    # LabelMapper.add("[SEP]")
    # LabelMapper.add("O")

    use_indices = [label > 3 for label in _flatten_list(golds)]
    metrics = {}
    if with_label:
        for mm in metric_meta:
            metric_name = mm.name
            metric_func = METRIC_FUNC[mm]
            metric = metric_func(
                np.array(_flatten_list(predictions))[use_indices],
                np.array(_flatten_list(golds))[use_indices]
            )
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids, inputs
