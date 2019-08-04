from data_utils.vocab import Vocabulary

NERLabelMapper = Vocabulary(True)
NERLabelMapper.add('O')
NERLabelMapper.add('B-MISC')
NERLabelMapper.add('I-MISC')
NERLabelMapper.add('B-PERSON')
NERLabelMapper.add('I-PERSON')
NERLabelMapper.add('B-ORGANIZATION')
NERLabelMapper.add('I-ORGANIZATION')
NERLabelMapper.add('B-LOCATION')
NERLabelMapper.add('I-LOCATION')
NERLabelMapper.add('B-ARTIFACT')
NERLabelMapper.add('I-ARTIFACT')
NERLabelMapper.add('B-DATE')
NERLabelMapper.add('I-DATE')
NERLabelMapper.add('B-TIME')
NERLabelMapper.add('I-TIME')
NERLabelMapper.add('B-MONEY')
NERLabelMapper.add('I-MONEY')
NERLabelMapper.add('B-PERCENT')
NERLabelMapper.add('I-PERCENT')
NERLabelMapper.add('X')
NERLabelMapper.add('[CLS]')
NERLabelMapper.add('[SEP]')

POSLabelMapper = Vocabulary(True)
POSLabelMapper.add("O")
POSLabelMapper.add("副詞")
POSLabelMapper.add("助詞")
POSLabelMapper.add("動詞")
POSLabelMapper.add("名詞")
POSLabelMapper.add("特殊")
POSLabelMapper.add("判定詞")
POSLabelMapper.add("助動詞")
POSLabelMapper.add("形容詞")
POSLabelMapper.add("感動詞")
POSLabelMapper.add("指示詞")
POSLabelMapper.add("接尾辞")
POSLabelMapper.add("接続詞")
POSLabelMapper.add("接頭辞")
POSLabelMapper.add("連体詞")
POSLabelMapper.add("未定義語")
POSLabelMapper.add("X")
POSLabelMapper.add("[CLS]")
POSLabelMapper.add("[SEP]")

GLOBAL_MAP = {
    'ner': NERLabelMapper,
    'pos': POSLabelMapper,
}

METRIC_META = {
    'ner': [7, 8, 9, 10, 11, 12],
    'pos': [7, 8, 9, 10, 11, 12],
}

SAN_META = {
    'ner': 2,
    'pos': 2,
}