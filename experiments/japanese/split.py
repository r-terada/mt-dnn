import os
import sys
from sklearn.model_selection import train_test_split


def read_file(fpath: str) -> list:
    '''
    return list of sentences
    '''
    sentences = []
    with open(fpath, 'r') as fp:
        lines = []
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue
            word = line.strip().split(' ')[0]
            if word != '。':
                lines.append(line)
            else:
                lines.append(line)
                if len(lines) > 1:
                    sentences.append(lines)
                lines = []
    return sentences

def main():
    fpath = sys.argv[1]
    sentences = read_file(fpath)
    print(f'contain {len(sentences)} sentences.')
    train, test = train_test_split(sentences, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.2)

    dir_path = os.path.dirname(fpath)

    print(f'train size: {len(train)}')
    train_path = os.path.join(dir_path, 'train.txt')
    print(f'write train examples to {train_path}')
    with open(train_path, 'w') as fp:
        for e in train:
            print('\n'.join(e), file=fp)
            print(file=fp)

    print(f'dev size: {len(dev)}')
    dev_path = os.path.join(dir_path, 'dev.txt')
    print(f'write dev examples to {dev_path}')
    with open(dev_path, 'w') as fp:
        for e in dev:
            print('\n'.join(e), file=fp)
            print(file=fp)

    print(f'test size: {len(test)}')
    test_path = os.path.join(dir_path, 'test.txt')
    print(f'write test examples to {test_path}')
    with open(test_path, 'w') as fp:
        for e in test:
            print('\n'.join(e), file=fp)
            print(file=fp)


if __name__ == "__main__":
    main()
    