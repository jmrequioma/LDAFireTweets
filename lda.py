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
spacy.load('en')
from spacy.lang.en import English
parser = English() 

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
	print(tweets.head())
	return tweets

def preprocess(tweets):
	tokenized_tweets = []
	# prepare stopwords
	stop_words = stopwords.words('english')

	# removing URLS
	tweets.text = tweets.text.apply(lambda x: re.sub(u'http\S+', u'', x))   

	# removing pic.twitter....
	tweets.text = tweets.text.apply(lambda x: re.sub(u'pic.twitter.com\S+', u'', x))

	# removing @... 
	tweets.text = tweets.text.apply(lambda x: re.sub(u'(\s)@\w+', u'', x))

	# removing hashtags
	tweets.text = tweets.text.apply(lambda x: re.sub(u'#', u'', x))

	# tokenize
	data_words = list(tweet_to_words(tweets.text))

	print(data_words)
	print(tweets.head())

	# remove stopwords
	data_words_nostops = remove_stopwords(data_words, stop_words)

	# Build the bigram and trigram models
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

	# Faster way to get a sentence clubbed as a trigram/bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)

	print(data_words_nostops)

def tweet_to_words(df_text):
	for tweet in df_text:
		yield(gensim.utils.simple_preprocess(str(tweet), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, model):
    return [model[doc] for doc in texts]

def make_trigrams(texts, model):
    return [model[bigram_mod[doc]] for doc in texts]

# def tokenize(tweets):
# 	lda_tokens = []
# 	tokens = parser(tweets)
# 	for token in tokens:
# 		if token.orth_.isspace():
# 			continue
# 		else:
# 			lda_tokens.append(token.lower_)
# 	return lda_tokens


def main():
	tweets = import_dataset()
	preprocess(tweets)

if __name__ == '__main__':
	main()