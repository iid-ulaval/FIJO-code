# CCF-dataset
Here is our code repo to reproduce the basic results of the article 
[“FIJO”: a French Insurance Soft Skill Detection Dataset](URL).

## Steps

1. Install Python dependencies uses by our codes using the following command

    ```bash
    pip install -Ur requirements.txt
    ```
2. Download our dataset using the following command

    ```bash
    python3 download_dataset
    ```
or by downloading it manually [here](URL) and unzipping it `experiment/data` directory.

3. Execute the following command 
    ```bash
    python3 init_other_dependencies.py
    ```
to download NLTK, FastText and CamemBERT models dependancies.

4. Execute the following command to generate our results
    ```bash
    python3 -m src.main
    ```