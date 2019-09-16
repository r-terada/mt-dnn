#! /bin/sh
python experiments/conll/conll_prepro.py
python prepro_std.py --model bert-base-uncased --root_dir data/canonical_data --task_def experiments/conll/conll_task_def.yml --do_lower_case $1
