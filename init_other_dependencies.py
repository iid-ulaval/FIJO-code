import os
import shutil

import fasttext.util
import nltk
import spacy

print("Starting of init of other dependencies.")

nltk.download('stopwords')
nltk.download('punkt')

spacy.cli.download("fr_core_news_md")

fasttext.util.download_model('fr', if_exists='ignore')
shutil.move("cc.fr.300.bin", "./experiment/embedding")
os.remove("cc.fr.300.bin.gz")

print("Init of other dependencies done.")
