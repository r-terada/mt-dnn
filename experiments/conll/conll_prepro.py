import os
import argparse
import random
from sys import path
path.append(os.getcwd())
from data_utils.log_wrapper import create_logger
from experiments.conll.conll_utils import dump_rows, load_ner

logger = create_logger(__name__, to_disk=True, log_file='conll_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing CoNLL dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')    
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # NER
    ######################################
    ner_train_path = os.path.join(root, 'conll2003/en/train.txt')
    ner_dev_path = os.path.join(root, 'conll2003/en/valid.txt')
    ner_test_path = os.path.join(root, 'conll2003/en/test.txt')

    ######################################
    # Loading DATA
    ######################################
    ner_train_data = load_ner(ner_train_path)
    ner_dev_data = load_ner(ner_dev_path)
    ner_test_data = load_ner(ner_test_path)
    logger.info('Loaded {} CoNLL2003 NER train samples'.format(len(ner_train_data)))
    logger.info('Loaded {} CoNLL2003 NER dev samples'.format(len(ner_dev_data)))
    logger.info('Loaded {} CoNLL2003 NER test samples'.format(len(ner_test_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    ner_train_fout = os.path.join(canonical_data_root, 'ner_train.tsv')
    ner_dev_fout = os.path.join(canonical_data_root, 'ner_dev.tsv')
    ner_test_fout = os.path.join(canonical_data_root, 'ner_test.tsv')
    dump_rows(ner_train_data, ner_train_fout)
    dump_rows(ner_dev_data, ner_dev_fout)
    dump_rows(ner_test_data, ner_test_fout)
    logger.info('done with ner')


if __name__ == '__main__':
    args = parse_args()
    main(args)
