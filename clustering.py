from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
# import mpld3
import json
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import logging
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import gensim
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.tag import pos_tag
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

data = pd.read_csv('tweets.csv')
data['text'] = data['text'].astype(str)
data_copy = data


def preprocessing(data):
    """
    Delete stopwords from NLTK and manual list of stopwords

    :param data: pandas data input
    :return: pandas without stopwords
    """
    data['text'] = data['text'].str.lower().str.split()

    # stop_words = stopwords.words('german')

    file = open('german_stopwords.txt', 'r')
    manual_stop_words = file.read()

    # remove stop words
    # data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in manual_stop_words])

    # remove special characters
    data['text'] = data['text'].apply(lambda x: [re.sub('[^A-Za-z0-9]+', '', item) for item in x])

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    data['text'] = data['text'].apply(lambda x: [tokenizer.tokenize(item) for item in x])

    data.columns = ['ID', 'user_screen_name', 'in_reply_to_screen_name', 'text', 'retweeted_screen_name', 'party']
    data = data.drop(['ID'], axis=1)

    # Uncomment following line to save it as csv
    # data_copy['text'].to_csv('cleaned_tweets_only.csv', header=True, sep=';')
    return list(data['text'])






def bagofwords(data):
    bag = []
    for tw in range(0, len(data['text'])):
        bag += data['text'][tw]
    return bag


data_cleaned = bagofwords(preprocessing(data))

#print(data_cleaned[33])
result = []
for i in range(0, len(data_cleaned)):
    try:
        result.append(data_cleaned[i][0])
    except:
        print('Empty list')
#print(result)

def read_corpus(data, tokens_only=False):
    for i, line in enumerate(data):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line)[i])


train_corpus = list(read_corpus(result, tokens_only=False))
print(train_corpus)


model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)


# model = Word2Vec(bagofwords(data), min_count=100)
words = list(model.wv.vocab)

X = model[model.wv.vocab]

num_clusters = 15
km = KMeans(n_clusters=num_clusters, random_state=42)

km.fit(X)

clusters = km.labels_().tolist()
print(clusters)
