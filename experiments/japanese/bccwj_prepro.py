import os
import sys
import json
import random
import argparse
from sys import path
path.append(os.getcwd())
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_utils.log_wrapper import create_logger
from experiments.japanese.bccwj_label_map import NERLabelMapper, NERALLLabelMapper, ChunkingLabelMapper, POSLabelMapper

logger = create_logger(__name__, to_disk=True, log_file='glue_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing BCCWJ dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--bert_model', type=str, default='mt_dnn_models/vocab.txt')
    parser.add_argument('--tasks', type=str, default='ner,pos')
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

            if len(words) > 0 and words[-1] == eos:
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


# TODO: input data が違うものが混ざっていて記憶力で task 指定してカバーしてるので厳しい
# ファイル分けるなりスクリプト中でファイル名指定するなりする
def load_nerall(file, eos='。'):
    """Loading data of bccwj with all ner labels
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

            if len(words) > 0 and words[-1] == eos:
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


def load_chunking(file, eos='。'):
    """Loading data of bccwj with all ner labels
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

            if len(words) > 0 and words[-1] == eos:
                sample = {'uid': str(cnt), 'premise': words, 'label': labels}
                rows.append(sample)
                cnt += 1
                words = []
                labels = []
            else:
                if len(contents.split(' ')) == 4:
                    word = contents.split(' ')[0]
                    label = contents.split(' ')[2]
                    words.append(word)
                    labels.append(label)
    return rows


def load_pos(file, eos='。'):
    """Loading data of bccwj with POS labels
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

            if len(words) > 0 and words[-1] == eos:
                sample = {'uid': str(cnt), 'premise': words, 'label': labels}
                rows.append(sample)
                cnt += 1
                words = []
                labels = []
            else:
                if len(contents.split(' ')) == 4:
                    word = contents.split(' ')[0]
                    label = contents.split(' ')[1]
                    words.append(word)
                    labels.append(label)
    return rows


def build_data(data, dump_path, tokenizer, label_mapper, max_seq_len=128):
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
            if len(tokens) > max_seq_len - 2:
                tokens = tokens[:max_seq_len - 2]
                labels = labels[:max_seq_len - 2]

            labels = ['[CLS]'] + labels[:max_seq_len - 2] + ['[SEP]']
            label = [label_mapper[lab] for lab in labels]
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            assert len(label) == len(input_ids)
            type_ids = [0] * ( len(tokens) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))


def main(args):
    root = args.data_dir
    assert os.path.exists(root)

    tasks = args.tasks.split(',')
    for t in tasks:
        if not t in ['ner', 'pos', 'nerall', 'chunking']:
            sys.exit(f'invalid task name: {t}')

    train_path = os.path.join(root, 'train.txt')
    dev_path = os.path.join(root, 'dev.txt')
    test_path = os.path.join(root, 'test.txt')

    tokenizer = BertTokenizer(args.bert_model, do_lower_case=False)

    ############
    # NER
    ############
    if 'ner' in tasks:
        # load
        train_data = load_ner(train_path)
        dev_data = load_ner(dev_path)
        test_data = load_ner(test_path)
        logger.info('Loaded {} NER train samples'.format(len(train_data)))
        logger.info('Loaded {} NER dev samples'.format(len(dev_data)))
        logger.info('Loaded {} NER test samples'.format(len(test_data)))

        # build
        train_fout = os.path.join(root, 'ner_train.json')
        dev_fout = os.path.join(root, 'ner_dev.json')
        test_fout = os.path.join(root, 'ner_test.json')

        build_data(train_data, train_fout, tokenizer, NERLabelMapper, args.max_seq_len)
        build_data(dev_data, dev_fout, tokenizer, NERLabelMapper, args.max_seq_len)
        build_data(test_data, test_fout, tokenizer, NERLabelMapper, args.max_seq_len)
        logger.info('done with NER')

    ############
    # NER ALL CLASS
    ############
    if 'nerall' in tasks:
        # load
        train_data = load_nerall(train_path)
        dev_data = load_nerall(dev_path)
        test_data = load_nerall(test_path)
        logger.info('Loaded {} NERALL train samples'.format(len(train_data)))
        logger.info('Loaded {} NERALL dev samples'.format(len(dev_data)))
        logger.info('Loaded {} NERALL test samples'.format(len(test_data)))

        # build
        train_fout = os.path.join(root, 'nerall_train.json')
        dev_fout = os.path.join(root, 'nerall_dev.json')
        test_fout = os.path.join(root, 'nerall_test.json')

        build_data(train_data, train_fout, tokenizer, NERALLLabelMapper, args.max_seq_len)
        build_data(dev_data, dev_fout, tokenizer, NERALLLabelMapper, args.max_seq_len)
        build_data(test_data, test_fout, tokenizer, NERALLLabelMapper, args.max_seq_len)
        logger.info('done with NERALL')

    ############
    # CHUNKING
    ############
    if 'chunking' in tasks:
        # load
        train_data = load_chunking(train_path)
        dev_data = load_chunking(dev_path)
        test_data = load_chunking(test_path)
        logger.info('Loaded {} Chunking train samples'.format(len(train_data)))
        logger.info('Loaded {} Chunking dev samples'.format(len(dev_data)))
        logger.info('Loaded {} Chunking test samples'.format(len(test_data)))

        # build
        train_fout = os.path.join(root, 'chunking_train.json')
        dev_fout = os.path.join(root, 'chunking_dev.json')
        test_fout = os.path.join(root, 'chunking_test.json')

        build_data(train_data, train_fout, tokenizer, ChunkingLabelMapper, args.max_seq_len)
        build_data(dev_data, dev_fout, tokenizer, ChunkingLabelMapper, args.max_seq_len)
        build_data(test_data, test_fout, tokenizer, ChunkingLabelMapper, args.max_seq_len)
        logger.info('done with Chunking')

    ############
    # POS Tagging
    ############
    if 'pos' in tasks:
        # load
        train_data = load_pos(train_path)
        dev_data = load_pos(dev_path)
        test_data = load_pos(test_path)
        logger.info('Loaded {} POS train samples'.format(len(train_data)))
        logger.info('Loaded {} POS dev samples'.format(len(dev_data)))
        logger.info('Loaded {} POS test samples'.format(len(test_data)))

        # build
        train_fout = os.path.join(root, 'pos_train.json')
        dev_fout = os.path.join(root, 'pos_dev.json')
        test_fout = os.path.join(root, 'pos_test.json')

        build_data(train_data, train_fout, tokenizer, POSLabelMapper, args.max_seq_len)
        build_data(dev_data, dev_fout, tokenizer, POSLabelMapper, args.max_seq_len)
        build_data(test_data, test_fout, tokenizer, POSLabelMapper, args.max_seq_len)
        logger.info('done with POSTagging')


if __name__ == "__main__":
    args = parse_args()
    main(args)
