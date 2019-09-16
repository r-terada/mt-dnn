import os
import argparse
import random
from sys import path
path.append(os.getcwd())
from data_utils.log_wrapper import create_logger
from experiments.conll.conll_utils import dump_rows, load_ner, load_chunking, load_pos

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

    train_path = os.path.join(root, 'conll2003/en/train.txt')
    dev_path = os.path.join(root, 'conll2003/en/valid.txt')
    test_path = os.path.join(root, 'conll2003/en/test.txt')

    ######################################
    # NER
    ######################################

    ner_train_data = load_ner(train_path)
    ner_dev_data = load_ner(dev_path)
    ner_test_data = load_ner(test_path)
    logger.info('Loaded {} CoNLL2003 NER train samples'.format(len(ner_train_data)))
    logger.info('Loaded {} CoNLL2003 NER dev samples'.format(len(ner_dev_data)))
    logger.info('Loaded {} CoNLL2003 NER test samples'.format(len(ner_test_data)))

    ######################################
    # Chunking
    ######################################

    chunking_train_data = load_chunking(train_path)
    chunking_dev_data = load_chunking(dev_path)
    chunking_test_data = load_chunking(test_path)
    logger.info('Loaded {} CoNLL2003 Chunking train samples'.format(len(chunking_train_data)))
    logger.info('Loaded {} CoNLL2003 Chunking dev samples'.format(len(chunking_dev_data)))
    logger.info('Loaded {} CoNLL2003 Chunking test samples'.format(len(chunking_test_data)))

    ######################################
    # Pos Tagging
    ######################################

    pos_train_data = load_pos(train_path)
    pos_dev_data = load_pos(dev_path)
    pos_test_data = load_pos(test_path)
    logger.info('Loaded {} CoNLL2003 POS Tagging train samples'.format(len(pos_train_data)))
    logger.info('Loaded {} CoNLL2003 POS Tagging dev samples'.format(len(pos_dev_data)))
    logger.info('Loaded {} CoNLL2003 POS Tagging test samples'.format(len(pos_test_data)))

    ######################################
    # DUMP DATA
    ######################################

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

    chunking_train_fout = os.path.join(canonical_data_root, 'chunking_train.tsv')
    chunking_dev_fout = os.path.join(canonical_data_root, 'chunking_dev.tsv')
    chunking_test_fout = os.path.join(canonical_data_root, 'chunking_test.tsv')
    dump_rows(chunking_train_data, chunking_train_fout)
    dump_rows(chunking_dev_data, chunking_dev_fout)
    dump_rows(chunking_test_data, chunking_test_fout)
    logger.info('done with chunking')

    pos_train_fout = os.path.join(canonical_data_root, 'pos_train.tsv')
    pos_dev_fout = os.path.join(canonical_data_root, 'pos_dev.tsv')
    pos_test_fout = os.path.join(canonical_data_root, 'pos_test.tsv')
    dump_rows(pos_train_data, pos_train_fout)
    dump_rows(pos_dev_data, pos_dev_fout)
    dump_rows(pos_test_data, pos_test_fout)
    logger.info('done with pos')


if __name__ == '__main__':
    args = parse_args()
    main(args)
