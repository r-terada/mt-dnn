import os
import sys
import json
from sys import path
path.append(os.getcwd())
from experiments.conll.conll_label_map import NERLabelMapper


if len(sys.argv) != 4:
    sys.exit('python format_conll_output.py raw_json_path orig_output_path new_output_path')

raw_json_path, orig_output_path, new_output_path = sys.argv[1:]

print(f'load raw_data from {raw_json_path}')
raw_data = []
with open(raw_json_path, 'r') as fp:
    for line in fp:
        raw_data.append(json.loads(line.strip()))
print(raw_data[:3])

print(f'load orig output from {orig_output_path}')
with open(orig_output_path, 'r') as fp:
    predictions = json.load(fp)['predictions']
print(predictions[:3])

print(f'write formatted output to {new_output_path}')
with open(new_output_path, 'w') as fp:
    for d, preds in zip(raw_data, predictions):
        tokens = d['token_id']
        labels = d['label']
        for t, l, p in zip(tokens, labels, preds):
            if int(l) == 0:
                continue
            l = NERLabelMapper.ind2tok(l)
            p = NERLabelMapper.ind2tok(p)
            print('\t'.join([t, l, p]), file=fp)

