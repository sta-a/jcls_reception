import os
import numpy as np
from spacy import load
os.chdir(os.path.dirname(__file__))
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.feature_extraction.text import CountVectorizer
# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(vectorizer.vocabulary_.items())
# id_map = dict((v, k) for k, v in vectorizer.vocabulary_.items())
# print(id_map)


class Pet(object):
   def __init__(self):
      self.myvar='bla'
   def my_method(self, a):
      print("I am a Cat", a)
      self.myvar = "new var"
cat = Pet()
cat.my_method('blabla')
print(cat.myvar)

dct1={'a':1, 'b':2, 'aa':0}
print(dct1.items())
print(sorted(list(dct1.items()), key=lambda x: x[1]))
most_common_k_ngrams = [ngram for ngram, count in sorted(list(dct1.items()), key=lambda x: x[1])[:3]]