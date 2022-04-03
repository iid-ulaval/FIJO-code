download-dataset: data-requirements
	python3 download_dataset.py

data-requirements:
	pip3 install wget

#######################################################################################################################################

reproduce-stats: stats-requirements
	cd stats; python3 stats.py

stats-requirements:
	pip3 install -r stats-requirements.txt

#######################################################################################################################################

device=0

reproduce-biLstm: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=lstm training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.01 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.logger.group_name="biLstm"

reproduce-camembertFrozen: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=True training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.01 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.logger.group_name="camembertFrozen"

reproduce-camembertUnfrozen: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=False training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.0001 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.logger.group_name="camembertUnfrozen"

reproduce-camembertUnfrozenWarmup: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=False training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.00002 \
		training.hyperparams.num_epochs=20 training.hyperparams.lr_warmup=True training.logs.logger.group_name="camembertUnfrozenWarmup"

experiment-requirements:
	pip3 install -r requirements.txt
	cd experiment/embedding;\
		python3 -c "import fasttext.util; fasttext.util.download_model('fr', if_exists='ignore')"
	rm -f experiment/embedding/cc.fr.300.bin.gz