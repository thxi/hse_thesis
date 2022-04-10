## Directory structure

- `bandits` - the main python package
- `notebooks` - jupyter notebooks with the experiments
- `data` - directory with datasets
  - `brazil` - the brazilian e-commerce dataset
  - `m5` - the m5 competition dataset
  - `avocado` - the avocado prices/demand dataset

## Installation

To install the project you need [poetry](https://python-poetry.org/)

After installing poetry run:

```sh
poetry install
```

To run a virtual environment with the installed packages:

```sh
poetry shell
```

## Datasets:

- sugar demand dataset is taken from https://users.stat.ufl.edu/~winner/data/
- Brazillian e-commerce https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Avocado prices https://www.kaggle.com/datasets/neuromusic/avocado-prices
- electricity prices https://www.openml.org/d/151

You can use [kaggle cli](https://www.kaggle.com/docs/api) to download the data.

Run

```sh
make data
```

to download all datasets
