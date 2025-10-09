setup:
	@echo "Setup virtual enviroment"
	@bash venv.sh

split:
	python mlp.py -s config/data.json data/data.csv

train:
	python mlp.py -t config/data.json data/train.csv

test:
	python mlp.py -p config/data.json data/test.csv params.json

digit:
	python mlp.py -s config/0-9.json data/0-9.csv
	python mlp.py -t config/0-9.json data/train.csv
	python mlp.py -p config/0-9.json data/test.csv params.json

regre-relu:
	python mlp.py -t config/regre-relu.json --gen-data1d
	python mlp.py -p config/regre-relu.json --gen-data1d params.json

regre-sigmoid:
	python mlp.py -t config/regre-sigmoid.json --gen-data1d
	python mlp.py -p config/regre-sigmoid.json --gen-data1d params.json