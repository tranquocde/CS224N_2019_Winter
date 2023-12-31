#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/sub_train.es --train-tgt=./en_es_data/sub_traintrain.en --dev-src=./en_es_data/sub_dev.es --dev-tgt=./en_es_data/sub_dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/sub_train.es --train-tgt=./en_es_data/sub_train.en --dev-src=./en_es_data/sub_dev.es --dev-tgt=./en_es_data/sub_dev.en --vocab=vocab.json --max-epoch=5
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/sub_test.es ./en_es_data/sub_test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
 