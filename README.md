# CCF-dataset
Here is our code repository to reproduce the basic results of the article 
[“FIJO”: a French Insurance Soft Skill Detection Dataset](URL).

The following details the steps necessary to fetch our dataset and reproduce our results.
Since each step might require multiple commands and/or command line arguments, we have put
a Makefile in place to ease the reproductibility experience.
## Steps

1. Download our dataset with the following command :
    ```bash
    make download-dataset
    ```

    or by downloading it manually [here](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP3/CHUEJM) and unzipping it `data/` directory
    at the root of the repository.

2. Once the dataset is downloaded, you can reproduce our dataset statistics as well as the results for each of our models in one simple command :

    - To reproduce dataset stats, run the following command :
         ```bash
        make reproduce-stats
        ``` 
    - To reproduce our bi-LSTM model results, run : 
        ```bash
        make reproduce-biLstm [device=0] [local_logging=False] [remote_logging=False]
        ``` 
    - To reproduce our CamemBERT frozen model results, run : 
        ```bash
        make reproduce-camembertFrozen [device=0] [local_logging=False] [remote_logging=False]
        ``` 
    - To reproduce our CamemBERT unfrozen model results, run : 
        ```bash
        make reproduce-camembertUnfrozen [device=0] [local_logging=False] [remote_logging=False]
        ``` 
    - To reproduce our CamemBERT unfrozen warmup model results, run : 
        ```bash
        make reproduce-camembertUnfrozenWarmup [device=0] [local_logging=False] [remote_logging=False]
        ```

    N.B: The last four commands include three optional arguments:

    - *device*: indicates which GPU device to use, if any. DEFAULT: 0
    - *local_logging*: boolean flag indicating whether or not to log model weights and metrics locally. Bear in mind that the CamemBERT based models have quite a high memory footprint. DEFAULT: False
    - *remote_logging*: boolean flag indicating whether or not to log model  metrics remotely using Weights & Biases. If *True*, you must be [logged to a Weights & Biases account locally](https://docs.wandb.ai/quickstart). DEFAULT: False

The installation of all the dependencies is handled automatically.

If you wish to run the python/pip commands manually, or if you're encountering problems with **make**, you can check out the annotated [Makefile](https://github.com/iid-ulaval/FIJO-code/blob/main/Makefile).
