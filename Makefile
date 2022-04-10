command:
	cat Makefile

jupyter:
	jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' --no-browser

lab:
	jupyter-lab --ip='*' --NotebookApp.token='' --NotebookApp.password='' --no-browser

data: data/m5 data/brazil data/avocado

data/m5: FORCE
	mkdir -p data/m5
	cd data/m5 && kaggle competitions download -c m5-forecasting-accuracy
	cd data/m5 && unzip -o  m5-forecasting-accuracy.zip

data/brazil: FORCE
	mkdir -p data/brazil
	cd data/brazil && kaggle datasets download -d olistbr/brazilian-ecommerce
	cd data/brazil && unzip -o brazilian-ecommerce.zip

data/avocado: FORCE
	mkdir -p data/avocado
	cd data/avocado && kaggle datasets download -d neuromusic/avocado-prices
	cd data/avocado && unzip -o avocado-prices.zip

FORCE: ;
