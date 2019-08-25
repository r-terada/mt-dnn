import os
import sys
import json
import argparse
from sys import path
path.append(os.getcwd())
from pytorch_pretrained_bert.tokenization import BertTokenizer

from experiments.japanese.bccwj_label_map import GLOBAL_MAP


def load(path):
    with open(path, 'r', encoding='utf-8') as reader:
        data = []
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
        return data


def main(args):
    label_map = GLOBAL_MAP[args.task_name]
    with open(args.scores_json_path, 'r', encoding='utf-8') as fp:
        scores = json.load(fp)
    data = load(args.data_json_path)

    tokenizer = BertTokenizer(args.bert_vocab_path, do_lower_case=False)

    with open(args.out_path, 'w') as fp:
        for d, predictions in zip(data, scores['predictions']):
            assert len(d['token_id']) == len(d['label'])
            assert len(d['label']) == len(predictions)
            tokens = [tokenizer.ids_to_tokens[t] for t in d['token_id']]
            labels = [label_map.ind2tok[l] for l in d['label']]
            preds = [label_map.ind2tok[l] for l in predictions]
            for t, l, p in zip(tokens, labels, preds):
                print(f"{t} {l} {p}", file=fp)
            print(file=fp)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        type=str,
                        required=True,
                        help="ner,nerall,pos,chunking")
    parser.add_argument("--bert_vocab_path",
                        type=str,
                        required=True,
                        help="path/to/bert/vocab.txt")
    parser.add_argument("--scores_json_path",
                        type=str,
                        required=True,
                        help="/path/to/<train,dev,test>_scores.json")
    parser.add_argument("--data_json_path",
                        type=str,
                        required=True,
                        help="/path/to/<train,dev,test>.json")
    parser.add_argument("--out_path",
                        type=str,
                        required=True,
                        help="/path/to/output/file")
    args = parser.parse_args()
    main(args)