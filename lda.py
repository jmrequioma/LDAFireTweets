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

def import_dataset():
	tweets = pd.read_csv('compiled.csv', sep=';', encoding = 'ISO-8859-1')
	print(tweets.head)
	return tweets

def preprocess(tweets):
	# prepare stopwords
	stop_words = stopwords.words('english')

	# removing URLS
	tweets.text = tweets.text.apply(lambda x: re.sub(u'http\S+', u'', x))   

	# removing @... 
	tweets.text = tweets.text.apply(lambda x: re.sub(u'(\s)@\w+', u'', x))

	# removing hashtags
	tweets.text = tweets.text.apply(lambda x: re.sub(u'#', u'', x))
	print(tweets.head())


def main():
	tweets = import_dataset()
	preprocess(tweets)

if __name__ == '__main__':
	main()