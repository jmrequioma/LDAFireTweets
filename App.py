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
matplotlib.use("TkAgg")   # for mac
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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
class App(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand = True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}

		frame = StartPage(container, self)

		self.frames[StartPage] = frame

		frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)

	def show_frame(self, cont):

		frame = self.frames[cont]
		frame.tkraise()

class StartPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		photo = PhotoImage(file="fire2.png")
		label = Label(self, image=photo)
		btn_browse = Button(self, text="Browse")

		label.place(anchor=CENTER)
		btn_browse.place(relx=0.5, rely=0.5, anchor=CENTER)
		
		
		

def main():
	app = App()
	app.configure(background='#DCDCDC')
	app.resizable(False, False)
	app.title("FireTalk Tweet Visualizer")
	app.geometry('600x500')
	app.mainloop()

if __name__ == '__main__':
	main()
