import re
import numpy as np
import pandas as pd
from pprint import pprint
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def prepare_stopwords():
	stop_words = stopwords.words('english')

def import_dataset():
	tweets = pd.read_csv('compiled.csv', sep=';', encoding = 'ISO-8859-1')
	print(tweets)

def main():
	import_dataset()
	prepare_stopwords()

if __name__ == '__main__':
	main()