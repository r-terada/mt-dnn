import os
import json
import random
import argparse
from sys import path
path.append(os.getcwd())
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_utils.log_wrapper import create_logger
from experiments.japanese.bccwj_label_map import NERLabelMapper

logger = create_logger(__name__, to_disk=True, log_file='glue_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing BCCWJ dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--bert_model', type=str, default='mt_dnn_models/bert_model_base_uncased.pt')
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()
    return args


def load_ner(file, eos='。'):
    """Loading data of bccwj with ner labels
    """
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            if contents.startswith("-DOCSTART-") or len(contents) == 0:
                    continue

            if words[-1] == eos:
                sample = {'uid': str(cnt), 'premise': words, 'label': labels}
                rows.append(sample)
                cnt += 1
                words = []
                labels = []
            else:
                word = contents.split(' ')[0]
                label = contents.split(' ')[-1]
                words.append(word)
                labels.append(label)
    return rows


def build_data(data, dump_path, tokenizer, max_seq_len=128):
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            mylabels = sample['label']
            tokens = []
            labels = []
            for i, word in enumerate(premise):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                for j in range(len(token)):
                    if j == 0:
                        labels.append(mylabels[i])
                    else:
                        labels.append('X')

            # Account for [CLS] and [SEP] with "- 2"
            if len(premise) > max_seq_len - 2:
                tokens = tokens[:max_seq_len - 2]
                labels = labels[:max_seq_len - 2]
            labels = ['[CLS]'] + labels[:max_seq_len - 2] + ['[SEP]']
            label = [NERLabelMapper[lab] for lab in labels]
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            assert len(label) == len(input_ids)
            type_ids = [0] * ( len(tokens) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))


def main(args):
    root = args.data_dir
    assert os.path.exists(root)

    train_path = os.path.join(root, 'train.txt')
    dev_path = os.path.join(root, 'dev.txt')
    test_path = os.path.join(root, 'test.txt')

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False, do_basic_tokenize=False)

    ############
    # NER
    ############

    # load
    train_data = load_ner(train_path)
    dev_data = load_ner(dev_path)
    test_data = load_ner(test_path)
    logger.info('Loaded {} NER train samples'.format(len(train_data)))
    logger.info('Loaded {} NER dev samples'.format(len(dev_data)))
    logger.info('Loaded {} NER test samples'.format(len(test_data)))

    # build
    train_fout = os.path.join(root, 'ner/train.json')
    dev_fout = os.path.join(root, 'ner/dev.json')
    test_fout = os.path.join(root, 'ner/test.json')

    build_data(train_data, train_fout, tokenizer, args.max_seq_len)
    build_data(dev_data, dev_fout, tokenizer, args.max_seq_len)
    build_data(test_data, test_fout, tokenizer, args.max_seq_len)
    logger.info('done with NER')


if __name__ == "__main__":
    args = parse_args()
    main(args)
