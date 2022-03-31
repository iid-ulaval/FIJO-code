reproduce-stats: stats-requirements
	cd stats
	python3 stats.py

stats-requirements:
	pip3 install stats-requirements.txt

###########################
# TODO: add logging option
# TODO: add device option

reproduce-biLstm: experiment-requirements
	cd experiment
	python3 -m src.main model=lstm training.hyperparams.initial_learning_rate=0.01 training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False

reproduce-camembertFrozen: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.initial_learning_rate=0.01 training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False

reproduce-camembertUnrozen: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.initial_learning_rate=0.0001 training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False

reproduce-camembertUnrozenWarmup: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.initial_learning_rate=0.00002 training.hyperparams.num_epochs=20 training.hyperparams.lr_warmup=True

experiment-requirements:
	pip3 install requirements.txt
	mkdir experiment/embedding
	cd experiment/embedding
	python3 -c "import fasttext.util; fasttext.util.download_model('fr', if_exists='ignore')"