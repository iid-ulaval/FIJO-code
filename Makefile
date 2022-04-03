# Commands to download the dataset
download-dataset: data-requirements
	python3 download_dataset.py

# Installation of requirements necessary to download the dataset
data-requirements:
	pip3 install wget

#######################################################################################################################################

# Commands to reproduce the statistics of the dataset
reproduce-stats: stats-requirements
	cd stats; python3 stats.py

# Installation of requirements necessary to reproduce dataset statistics
stats-requirements:
	pip3 install -r stats-requirements.txt

#######################################################################################################################################

# GPU device to use. Default: 0
device=0

# Whether to log the weights and results locally: Default: False
local_logging=False

# Whether to log the weights and results remotely on Weights & Biases: Default: False
remote_logging=False

# Commands to reproduce the bi-LSTM results
reproduce-biLstm: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=lstm training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.01 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.local.logging=$(local_logging) training.logs.logger.logging=$(remote_logging) \
		training.logs.logger.group_name="biLstm"

# Commands to reproduce the CamemBERT frozen results
reproduce-camembertFrozen: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=True training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.01 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.local.logging=$(local_logging) training.logs.logger.logging=$(remote_logging) \
		training.logs.logger.group_name="camembertFrozen"

# Commands to reproduce the CamemBERT unfrozen results
reproduce-camembertUnfrozen: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=False training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.0001 \
		training.hyperparams.num_epochs=300 training.hyperparams.lr_warmup=False training.logs.local.logging=$(local_logging) training.logs.logger.logging=$(remote_logging) \
		training.logs.logger.group_name="camembertUnfrozen"

# Commands to reproduce the CamemBERT unfrozen warmup results
reproduce-camembertUnfrozenWarmup: experiment-requirements
	cd experiment; \
		HYDRA_FULL_ERROR=1 python3 -m src.main model=camembert model.freeze_camembert=False training.hyperparams.train_device_id=$(device) training.hyperparams.initial_learning_rate=0.00002 \
		training.hyperparams.num_epochs=20 training.hyperparams.lr_warmup=True training.logs.local.logging=$(local_logging) training.logs.logger.logging=$(remote_logging) \
		training.logs.logger.group_name="camembertUnfrozenWarmup"

# Installation of requirements necessary to run the experiments and reproduce the results
experiment-requirements:
	pip3 install -r requirements.txt
	cd experiment/embedding;\
		python3 -c "import fasttext.util; fasttext.util.download_model('fr', if_exists='ignore')"
	rm -f experiment/embedding/cc.fr.300.bin.gz