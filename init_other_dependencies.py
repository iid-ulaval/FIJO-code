import shutil

import nltk
import spacy
import fasttext.util

print("Starting of init of other dependencies.")

nltk.download('stopwords')
nltk.download('punkt')

spacy.cli.download("fr_core_news_md")

fasttext.util.download_model('fr', if_exists='ignore')
shutil.move("cc.fr.300.bin", "./experiment/embedding")

print("Init of other dependencies done.")
