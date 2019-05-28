import csv
import re
import math
import numpy as np
import pandas as pd
from pprint import pprint
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from collections import defaultdict
from nltk.corpus import words
word_list = words.words()
import threading
import queue
import time
import uuid
import uuid
from time import gmtime, strftime
from tkinter.ttk import Progressbar

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

# textblob for spellchecking
from textblob import TextBlob
from textblob import Word

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim
# pyLDAvis.enable_notebook()
import matplotlib
# matplotlib.use("TkAgg")   # for mac
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

LARGE_FONT = ("Verdana", 12)

# DISP_INTEGRITYS = []
# DISP_SP_ENTROPYS = []
# DISP_TOPIC_WEIGHTS = []
# DISP_TC_SCORE = 0

FILENAME = ""
class App(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.resizable(False, False)
		self.title("FireTalk Tweet Visualizer")
		self.geometry('600x500')
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand = True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}

		for F in (StartPage, LoadingPage, MetricsPage, TopicPage):
			frame = F(container, self)
			self.frames[F] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)

	def show_frame(self, cont):

		frame = self.frames[cont]
		frame.tkraise()

class StartPage(tk.Frame):
	file = 'this is file'
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		photo = PhotoImage(file="fire2.png")
		label = Label(self, image=photo)
		btn_browse = Button(self, text="Browse", command=lambda: self.open_file(controller))

		label.place(anchor=CENTER)
		label.image = photo
		btn_browse.place(relx=0.5, rely=0.5, anchor=CENTER)
		

	def open_file(self, controller):
		global FILENAME
		FILENAME = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
		controller.show_frame(LoadingPage)

		
class LoadingPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.label = Label(self, text="")
		self.test_button = Button(self, text="Run", command=lambda: self.holder(controller))
		self.home_button = Button(self, text="Go to Metrics", command=lambda: [controller.show_frame(MetricsPage), self.reset(controller)])
		self.barVar = tk.DoubleVar()
		self.barVar.set(0)
		self.bar = Progressbar(self, length=400, variable=self.barVar)


		self.label.place(x=260, y=220)
		self.bar.place(x=100, y=240)
		self.test_button.place(x=260, y=270)
		self.home_button.pack(side=BOTTOM)

	def logic(self, controller):
		try:
			# print(StartPage.file)
			print(FILENAME)
			tweets = import_dataset(FILENAME)
			data_lemmatized = preprocess(tweets)
			lda(data_lemmatized)
		except Exception as e:
			print(FILENAME)
			raise e
			print("error")
			controller.show_frame(StartPage)

	def holder(self, controller):
		t1 = threading.Thread(target=self.logic, args=(controller, ))
		t1.start()
		t2 = threading.Thread(target=self.update_progressbar, args=(controller, ))
		t2.start()
		self.test_button.config(state=DISABLED)
		self.home_button.config(state=DISABLED)

	def update_progressbar(self, controller):
		print("test")
		x = self.barVar.get()
		if x < 100:
			time.sleep(1)
			if (x <= 25):
				self.label.place(x=240, y=220)
				self.label['text'] = 'Removing Stopwords...'
			elif (x > 25 and x <= 50):
				self.label.place(x=220, y=220)
				self.label['text'] = 'Correcting misspelled words...'
			elif (x > 50 and x <= 60):
				self.label.place(x=260, y=220)
				self.label['text'] = 'Lemmatizing...'
			elif (x > 60):
				self.label.place(x=260, y=220)
				self.label['text'] = 'Running LDA...'

			self.barVar.set(x+0.5)
			self.update_progressbar(controller)
		else:
			print("Complete")
			self.label.place(x=265, y=220)
			self.label['text'] = 'Finalizing...'
			self.home_button.config(state=NORMAL)

	def reset(self, controller):
		self.label['text'] = ''
		self.barVar.set(0)
		self.test_button.config(state=NORMAL)


class MetricsPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		home_button = Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
		metrics_button = Button(self, text="Get Metrics", command=lambda: self.get_metrics(controller))
		integrity_lbl = Label(self, text="Integrity Scores")
		self.integrity_val = Label(self, text="")
		sp_lbl = Label(self, text="Spatial Entropy Scores")
		self.sp_val = Label(self, text="")
		tw_lbl = Label(self, text="Topic Weights")
		self.tw_val = Label(self, text="")
		tc_lbl = Label(self, text="Topic Coherence Score")
		self.tc_val = Label(self, text="")


		integrity_lbl.place(x=100, y=100)
		self.integrity_val.place(x=100, y=120)
		sp_lbl.place(x=100, y=140)
		self.sp_val.place(x=100, y=160)
		tw_lbl.place(x=100, y=180)
		self.tw_val.place(x=100, y=200)
		tc_lbl.place(x=100, y=220)
		self.tc_val.place(x=100, y=240)
		metrics_button.place(x=260, y=300)
		home_button.pack(side=BOTTOM)

	def get_metrics(self, controller):
		results = pd.read_csv('output_scores.csv', sep=',', skip_blank_lines=True, encoding = 'ISO-8859-1')
		results_arrs = results.values.tolist()
		data_row = results_arrs[0]
		disp_integritys = data_row[1]
		disp_sp_entropys = data_row[2]
		disp_tw = data_row[3]
		disp_tc = data_row[4]
		self.integrity_val['text'] = str(disp_integritys)
		self.sp_val['text'] = str(disp_sp_entropys)
		self.tw_val['text'] = str(disp_tw)
		self.tc_val['text'] = str(disp_tc)


class TopicPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		label = Label(self, text="Loading...")
		home_button = Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
		coherence_button = Button(self, text="Back to Coherence", command=lambda: controller.show_frame(MetricsPage))

		label.place(relx=0.5, rely=0.5, anchor=CENTER)
		home_button.pack(side=BOTTOM)
		coherence_button.pack(side=BOTTOM)

def logic(filename):
	try:
		tweets = import_dataset(filename)
		data_lemmatized = preprocess(tweets)
		lda(data_lemmatized)
	except Exception as e:
		# raise e
		print("error")

def import_dataset(csv_file):
	try:
		tweets = pd.read_csv(csv_file, sep=';', encoding = 'ISO-8859-1')
		print(tweets.head())
		return tweets
	except Exception as e:
		# raise e
		print("error2")

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

	# spellcheck
	for data_word in data_words_nostops:
		i = 0
		for word in data_word:
			w = Word(word)
			list_of_spelled = w.spellcheck()   # list of tuples -> [(word, confidence), ...]
			res = [lis[0] for lis in list_of_spelled]
			spelled_word = res[0]
			conf = [lis[1] for lis in list_of_spelled]
			if (conf[0] > 0 and conf[0] < 0.8):
				print(w.spellcheck())
			else:
				data_word[i] = spelled_word
			i = i + 1

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
	doc_lda = lda_model[corpus]   # get topic probability distribution for a document

	# Compute Perplexity
	print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

	# Compute Coherence Score
	coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()
	print('\nCoherence Score: ', coherence_lda)


	model_list, coherence_values = compute_coherence_values(lda_model, dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=15, step=1)
	COHERENCE_VALUES = coherence_values
	# mallet_path = '/Users/student/Downloads/mallet-2.0.8/bin/mallet'
	# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=8, id2word=id2word)
	# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
	# print('\nCoherence Score: ', coherence_ldamallet)

	# Show graph
	limit=15; start=2; step=1;
	x = range(start, limit, step)
	fig = plt.figure()
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.legend(("coherence_values"), loc='best')
	# plt.show()
	unique_filename = str(uuid.uuid4())
	plt.savefig('graphs/TC'+ unique_filename+ '.png')
	count = 0
	temp = 0
	ideal_num_topics = 0
	# Print the coherence scores
	for m, cv in zip(x, coherence_values):
		if (count == 0):
			temp = round(cv, 4)
		else:
			# count > 0
			if (round(cv, 4) > temp):
				ideal_num_topics = m
				print(ideal_num_topics)
				temp = round(cv, 4)
				print(temp)
		count = count + 1

	if (ideal_num_topics == 0):
		for m, cv in zip(x, coherence_values):
			if (count == 0):
				temp = round(cv, 4)
			else:
				# count > 0
				if (round(cv, 4) < temp and m > 3):
					ideal_num_topics = m - 1
					print(ideal_num_topics)
					temp = round(cv, 4)
					print(temp)
					break
				temp = round(cv, 4)
		count = count + 1
	print(ideal_num_topics)
	half_of_topics = int(ideal_num_topics / 2)
	# if (ideal_num_topics % 2 == 0):
	# 	half_of_topics = int(ideal_num_topics / 2)
	# else:
	# 	ideal_num_topics = ideal_num_topics - 1
	# 	half_of_topics = int((ideal_num_topics) / 2)

	# feed the lda model with ideal number of topics 
	real_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=ideal_num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

	topics = real_lda_model.show_topics(num_topics = ideal_num_topics, formatted=False)
	data_flat = [w for w_list in texts for w in w_list]
	counter = Counter(data_flat)
	doc_lda = real_lda_model[corpus]

	coherence_model_lda = CoherenceModel(model=real_lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
	coherence_lda = coherence_model_lda.get_coherence()

	print('\nActual Coherence Score: ', coherence_lda)
	DISP_TC_SCORE = coherence_lda
	out = []
	integritys = []   # integrity of topic
	dists = []
	integrity = 0
	for i, topic in topics:
		integrity = 0
		for word, weight in topic:
			out.append([word, i , weight, counter[word]])


	print("data_flat!!")
	print(data_flat)
	topic_counter = 0
	prob_of_word = 0
	testDict = defaultdict(float)
	for w in data_flat:
		list_of_probs = real_lda_model.get_term_topics(w)
		for topic_num, prob_value in list_of_probs:
			testDict[topic_num] += prob_value
		topic_counter = topic_counter + 1

	print(sorted(testDict.items()))
	integritys = [i[1] for i in sorted(testDict.items())]
	print(integritys)

	integrity_mean = np.mean(integritys)
	integry_std = np.std(integritys)
	# check
	if (len(integritys) < ideal_num_topics):
		for i in range(ideal_num_topics - len(integritys)):
			integritys.append(0)
	# normalize integrity
	normalized_integritys = []
	normalized_integrity = 0
	for i in range(len(integritys)):
		normalized_integrity = (integritys[i] - integrity_mean) / integry_std
		normalized_integritys.append(round(normalized_integrity, 4))
	print("integrity mean: ")
	print(integrity_mean)
	print(integry_std)
	print("normalized_integritys")
	print(normalized_integritys)
	DISP_INTEGRITYS = normalized_integritys
	for topic in doc_lda:
		# print(topic)
		for topic_num, prob in topic[0]:
			dists.append([topic_num, prob, math.log10(prob), prob * math.log10(prob)])

	df_for_word_doc = pd.DataFrame(dists, columns=['topic_num', 'prob', 'prob_log', 'mult_prob_log'])
	print(df_for_word_doc)
	sp_entropys = df_for_word_doc.groupby('topic_num').sum().mult_prob_log
	# sp_mean = df_for_word_doc.loc[:, "topic_num"].mean()
	# print(sp_mean)
	negated_sp = (sp_entropys * -1).tolist()
	sp_mean = np.mean(negated_sp)
	print("mean")
	print(sp_mean)
	sp_std = np.std(negated_sp)
	print("negated_sp")
	print(negated_sp)
	print("sp_std")
	print(sp_std)

	# normalize spatial entropy
	normalized_sp_entropys = []
	normalized_sp_entropy = 0
	for i in range(len(negated_sp)):
		normalized_sp_entropy = (negated_sp[i] - sp_mean) / sp_std
		normalized_sp_entropys.append(round(normalized_sp_entropy, 4))

	print("normalized_sp_entropys")
	print(normalized_sp_entropys)
	DISP_SP_ENTROPYS = normalized_sp_entropys
	# check length
	if (len(normalized_sp_entropys) < ideal_num_topics):
		for i in range(ideal_num_topics - len(normalized_sp_entropys)):
			normalized_sp_entropys.append(0)

	# calculate weights of topics
	topic_weights = []
	topic_weight = 0
	for i in range(len(integritys)):
		topic_weight = normalized_integritys[i] - normalized_sp_entropys[i]
		topic_weights.append(round(topic_weight, 4))

	print("topic weights!!!!")
	print(topic_weights)
	DISP_TOPIC_WEIGHTS = topic_weights

	with open('output_scores.csv', mode='w') as output_file:
		output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		output_writer.writerow(['Number of Topics', 'Integritys', 'Spatial Entropy', 'Topic Weights', 'Coherence Score'])
		output_writer.writerow([len(topic_weights), normalized_integritys, normalized_sp_entropys, topic_weights, round(coherence_lda, 4)])
	# print(integritys)


	df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
	print(df)

	# Plot Word Count and Weights of Topic Keywords
	# plt.figure()
	fig, axes = plt.subplots(half_of_topics, 2, figsize=(10,10), sharey=True, dpi=90, squeeze=True)

	cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
	counter = 0
	df_word_list = df["word"].tolist()
	df_wordcount_list = df["word_count"].tolist()
	df_wordweight_list = df["importance"].tolist()
	start = 0
	color_iter = 0
	print(df_word_list)
	if (ideal_num_topics % 2 == 0):
		for i, ax in enumerate(axes.flatten()):
			
			sliced_word_list = df_word_list[start:start + 10]
			sliced_wordcount_list = df_wordcount_list[start:start + 10]
			sliced_wordweight_list = df_wordweight_list[start:start + 10]

			if (color_iter == 10):
				color_iter = 0

			x1 = np.arange(10)
			print(i)
			ax.bar(x=x1, height=sliced_wordcount_list, color=cols[color_iter], width=0.5, alpha=0.3, label='Word Count')
			ax.set_ylabel('Word Count', color=cols[color_iter])
			ax_twin = ax.twinx()
			ax_twin.bar(x=x1, height=sliced_wordweight_list, color=cols[color_iter], width=0.2, label='Weights')
			ax.set_title('Topic: ' + str(i), color=cols[color_iter], fontsize=10)
			ax.tick_params(axis='y', left=False)
			ax.set_xticks(np.arange(len(sliced_word_list)))
			ax.set_xticklabels(sliced_word_list, rotation=30, horizontalalignment= 'right')
			ax.legend(loc='upper left')
			ax_twin.legend(loc='upper right')
			start = start + 10
			color_iter = color_iter + 1

		fig.tight_layout(w_pad=2)    
		fig.suptitle('Topic Weights: ' + str(topic_weights), fontsize='10')
		# plt.show()
		unique_filename = str(uuid.uuid4())
		plt.savefig('graphs/T0'+unique_filename+'.png')
	else:
		for i, ax in enumerate(axes.flatten()):
			sliced_word_list = df_word_list[start:start + 10]
			sliced_wordcount_list = df_wordcount_list[start:start + 10]
			sliced_wordweight_list = df_wordweight_list[start:start + 10]

			if (color_iter == 10):
				color_iter = 0

			x1 = np.arange(10)
			ax.bar(x=x1, height=sliced_wordcount_list, color=cols[color_iter], width=0.5, alpha=0.3, label='Word Count')
			ax.set_ylabel('Word Count', color=cols[color_iter])
			ax_twin = ax.twinx()
			ax_twin.bar(x=x1, height=sliced_wordweight_list, color=cols[color_iter], width=0.2, label='Weights')
			ax.set_title('Topic: ' + str(i), color=cols[color_iter], fontsize=10)
			ax.tick_params(axis='y', left=False)
			ax.set_xticks(np.arange(len(sliced_word_list)))
			ax.set_xticklabels(sliced_word_list, rotation=30, horizontalalignment= 'right')
			ax.legend(loc='upper left')
			ax_twin.legend(loc='upper right')
			start = start + 10
			color_iter = color_iter + 1

		fig.tight_layout(w_pad=2)    
		fig.suptitle('Topic Weights: ' + str(topic_weights), fontsize=10)    
		# plt.show()
		unique_filename = str(uuid.uuid4())
		plt.savefig('graphs/T1'+unique_filename+'.png')
		# plt.figure()
		fig, axes = plt.subplots(1, 1, sharey=True, dpi=90, squeeze=True)

		x1 = np.arange(10)
		axes.bar(x=x1, height=df_wordcount_list[start:start+10], color=cols[0], width=0.5, alpha=0.3, label='Word Count')
		axes.set_ylabel('Word Count', color=cols[0])
		ax_twin = axes.twinx()
		ax_twin.bar(x=x1, height=df_wordweight_list[start:start+10], color=cols[0], width=0.2, label='Weights')
		axes.set_title('Topic: ' + str(len(topics) - 1), color=cols[0], fontsize=10)
		axes.tick_params(axis='y', left=False)
		axes.set_xticks(np.arange(len(df_word_list[start:start+10])))
		axes.set_xticklabels(df_word_list[start:start+10], rotation=30, horizontalalignment= 'right')
		axes.legend(loc='upper left')
		ax_twin.legend(loc='upper right')
		# start = start + 10

		fig.tight_layout(w_pad=2)    
		fig.suptitle('Cont.', fontsize='10')    
		# plt.show()
		unique_filename = str(uuid.uuid4())
		plt.savefig('graphs/T2'+unique_filename+'.png')






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
	app = App()
	# app.configure(background='#DCDCDC')
	# app.resizable(False, False)
	# app.title("FireTalk Tweet Visualizer")
	# app.geometry('600x500')

	app.mainloop()

if __name__ == '__main__':
	main()
