reproduce-stats: stats-requirements
	cd stats; python3 stats.py

stats-requirements:
	pip3 install -r stats-requirements.txt

#######################################################################################################################################

device=0

reproduce-biLstm: experiment-requirements
	cd experiment
	python3 -m src.main model=lstm training.hyperparams.device=$(device) training.hyperparams.initial_learning_rate=0.01 \ 
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False  training.logs.logger.group_name="biLstm"

reproduce-camembertFrozen: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.device=$(device) training.hyperparams.initial_learning_rate=0.01 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.logger.group_name="camembertFrozen"

reproduce-camembertUnrozen: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.device=$(device) training.hyperparams.initial_learning_rate=0.0001 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.logger.group_name="camembertUnfrozen"

reproduce-camembertUnrozenWarmup: experiment-requirements
	cd experiment
	python3 -m src.main model=camembert training.hyperparams.device=$(device) training.hyperparams.initial_learning_rate=0.00002 \
		training.hyperparams.num_epochs=20 training.hyperparams.lr_warmup=True training.logs.logger.group_name="camembertUnfrozenWarmup"

experiment-requirements:
	pip3 install -r requirements.txt
	mkdir experiment/embedding
	cd experiment/embedding
	python3 -c "import fasttext.util; fasttext.util.download_model('fr', if_exists='ignore')"