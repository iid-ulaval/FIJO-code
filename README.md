# CCF-dataset
Code used to generate basic result for published CCF dataset

After installing of python environment with requirements.txt:
1. Copy data in experiment/data
2. Execute script `init_ccf.py` to download fasttext and CamemBERT pretrained models
3. Put fasttext embeddings in experiment/embedding
4. Logging needs Weight&Biases account
5. in experiment/ launch with `python3 -m src.main`