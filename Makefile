setup:
	@echo "Setup virtual enviroment"
	uv sync

build:
	uv build

clean:
	rm -rf ./dist \
			./.venv

fclean: clean
	rm -rf params.json predictions.json visualize
	
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

split:
	uv run -m mlp -s config/data.json data/data.csv

train:
	uv run -m mlp -t config/data.json data/train.csv

test:
	uv run -m mlp -p config/data.json data/test.csv params.json

digit:
	uv run -m mlp -s config/0-9.json data/0-9.csv
	uv run -m mlp -t config/0-9.json data/train.csv
	uv run -m mlp -p config/0-9.json data/test.csv params.json

regre-relu:
	uv run -m mlp -t config/regre-relu.json --gen-data1d
	uv run -m mlp -p config/regre-relu.json --gen-data1d params.json

regre-sigmoid:
	uv run -m mlp -t config/regre-sigmoid.json --gen-data1d
	uv run -m mlp -p config/regre-sigmoid.json --gen-data1d params.json
