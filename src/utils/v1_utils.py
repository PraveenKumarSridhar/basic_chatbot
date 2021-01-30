import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
stemmer = LancasterStemmer()

def get_bow(sent,vocab):
    s_words = nltk.word_tokenize(sent)
    s_words = [stemmer.stem(w.lower()) for w in s_words]
    bow = [1 if w in s_words else 0 for w in vocab]
    return np.array(bow)