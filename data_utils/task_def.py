# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    SequenceLabeling = 4


class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
