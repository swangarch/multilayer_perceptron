#!/bin/bash

echo "-------------------------------------------TEST for 1st config----------------------------------------------------------------------------"

python mlp.py -s config/data-sigmoid.json data/data.csv

python mlp.py -t config/data-sigmoid.json data/train.csv

python mlp.py -p config/data-sigmoid.json data/test.csv params.json


echo "-------------------------------------------TEST for 2nd config----------------------------------------------------------------------------"

python mlp.py -s config/data-softmax.json data/data.csv

python mlp.py -t config/data-softmax.json data/train.csv

python mlp.py -p config/data-softmax.json data/test.csv params.json


echo "-------------------------------------------TEST for 3nd config----------------------------------------------------------------------------"

python mlp.py -s config/0-1.json data/0-1.csv

python mlp.py -t config/0-1.json data/train.csv

python mlp.py -p config/0-1.json data/test.csv params.json


echo "-------------------------------------------TEST for 4th config----------------------------------------------------------------------------"

python mlp.py -t config/regre-relu.json --gen-data1d

python mlp.py -p config/regre-relu.json --gen-data1d params.json


echo "-------------------------------------------TEST for 5th config----------------------------------------------------------------------------"

python mlp.py -t config/regre-sigmoid.json --gen-data1d

python mlp.py -p config/regre-sigmoid.json --gen-data1d params.json


echo "-------------------------------------------TEST for 6th config----------------------------------------------------------------------------"

python mlp.py -s config/0-9.json data/0-9.csv

python mlp.py -t config/0-9.json data/train.csv

python mlp.py -p config/0-9.json data/test.csv params.json


