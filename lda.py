import re
import numpy as np
import pandas as pd
from pprint import pprint
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
# from spacy.lang.en import English
# parser = English() 

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from tkinter import *
from tkinter.ttk import *

def import_dataset():
	tweets = pd.read_csv('pasil.csv', sep=';', encoding = 'ISO-8859-1')
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
	tweets.text = tweets.text.apply(lambda x: re.sub(u'#\S+', u'', x))

	# tokenize
	data_words = list(tweet_to_words(tweets.text))

	print(data_words)
	print(tweets.head())

	# Build the bigram and trigram models
	bigram = gensim.models.Phrases(data_words, min_count=3, threshold=30) # higher threshold fewer phrases.
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

	# Faster way to get a sentence clubbed as a trigram/bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)

	# remove stopwords
	data_words_nostops = remove_stopwords(data_words, stop_words)

	# form bigrams
	data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

	#lemmatize
	data_words_bigrams_lemmas = get_lemma(data_words_bigrams)

	# Do lemmatization keeping only noun, adj, vb, adv
	data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

	print(data_lemmatized)
	return data_lemmatized

def tweet_to_words(df_text):
	for tweet in df_text:
		#yield(tokenize(str(tweet)))
		yield(gensim.utils.simple_preprocess(str(tweet), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, model):
    return [model[doc] for doc in texts]

def make_trigrams(texts, model):
    return [model[bigram_mod[doc]] for doc in texts]

def get_lemma(texts):
	return [WordNetLemmatizer().lemmatize(str(doc)) for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# def tokenize(tweets):
# 	lda_tokens = []
# 	tokens = parser(tweets)
# 	for token in tokens:
# 		if token.orth_.isspace():
# 			continue
# 		else:
# 			lda_tokens.append(token.lower_)
# 	return lda_tokens

def get_id2word(data_lemmatized):
	return corpora.Dictionary(data_lemmatized)

def lda(data_lemmatized):
	# Create Dictionary
	id2word = corpora.Dictionary(data_lemmatized)

	# Create Corpus
	texts = data_lemmatized

	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in texts]

	# Build LDA model
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

	pprint(lda_model.print_topics())
	doc_lda = lda_model[corpus]
	# Compute Perplexity
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

	# Compute Coherence Score
	coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)


	model_list, coherence_values = compute_coherence_values(lda_model, dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=15, step=3)

	# mallet_path = '/Users/student/Downloads/mallet-2.0.8/bin/mallet'
	# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=8, id2word=id2word)
	# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
	# print('\nCoherence Score: ', coherence_ldamallet)

	# Show graph
	limit=1000; start=2; step=1;
	x = range(start, limit, step)
	# plt.plot(x, coherence_values)
	# plt.xlabel("Num Topics")
	# plt.ylabel("Coherence score")
	# plt.legend(("coherence_values"), loc='best')
	# plt.show()
	count = 0
	temp = 0
	# Print the coherence scores
	for m, cv in zip(x, coherence_values):
		if (count == 0):
			temp = round(cv, 4)
		else:
			# count > 0
			if (round(cv, 4) < temp):
				ideal_num_topics = m
				print(ideal_num_topics)
				break
			else:
				temp = round(cv, 4)
		count = count + 1

	half_of_topics = 0
	if (ideal_num_topics % 2 == 0):
		half_of_topics = int(ideal_num_topics / 2)
	else:
		half_of_topics = int((ideal_num_topics - 1) / 2)

	# feed the lda model with ideal number of topics 
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=ideal_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

	topics = lda_model.show_topics(formatted=False)
	data_flat = [w for w_list in texts for w in w_list]
	counter = Counter(data_flat)

	out = []
	for i, topic in topics:
	    for word, weight in topic:
	        out.append([word, i , weight, counter[word]])

	df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

	# Plot Word Count and Weights of Topic Keywords
	fig, axes = plt.subplots(half_of_topics, 2, figsize=(10,10), sharey=True, dpi=90, squeeze=True)
	cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
	for i, ax in enumerate(axes.flatten()):
	    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
	    ax_twin = ax.twinx()
	    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
	    ax.set_ylabel('Word Count', color=cols[i])
	    ax_twin.set_ylim(0, 0.1); ax.set_ylim(0, 3500)
	    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
	    ax.tick_params(axis='y', left=False)
	    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
	    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

	fig.tight_layout(w_pad=2)    
	fig.suptitle('Word Count and Importance of Topic Keywords')    
	plt.show()

	# find for best coherence value score

def compute_coherence_values(model, dictionary, corpus, texts, limit, start=2, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    mallet_path = '/Users/student/Downloads/mallet-2.0.8/bin/mallet'
    id2word = get_id2word(texts)
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
    	# for mallet
        # model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        # model_list.append(model)
        # coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        # coherence_values.append(coherencemodel.get_coherence())

        # lda
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherence_model_lda = CoherenceModel(model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_lda)


    return model_list, coherence_values

def main():
	window = Tk()
	window.configure(background='lightgray')
	window.resizable(False, False)
	window.title("FireTalk Tweet Visualizer")
	window.geometry('600x500')
	photo = PhotoImage(file="fire2.png")
	label = Label(window, image=photo)
	tweets = import_dataset()
	data_lemmatized = preprocess(tweets)
	btn = Button(window, text="Visualize", command= lambda: lda(data_lemmatized))

	label.place(anchor=CENTER)
	btn.place(relx=0.5, rely=0.5, anchor=CENTER)
	window.mainloop()

if __name__ == '__main__':
	main()