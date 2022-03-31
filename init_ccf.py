import spacy
import fasttext.util

fasttext.util.download_model('fr', if_exists='ignore')
spacy.cli.download("fr_core_news_lg")
