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
    return data['text']



def bagofwords(data):
    bag = []
    for tw in range(0, len(data)):
        bag += data[tw]
    return bag


data_c = bagofwords(preprocessing(data))

#data_cleaned = bagofwords(bagofwords(preprocessing(data)))


dictionary = corpora.Dictionary(data_c)
dictionary.filter_extremes(no_below=5, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in data_c]

#print(corpus)

lda = models.LdaMulticore(corpus, num_topics=15, workers=40, id2word=dictionary, chunksize=10000, passes=1)

topics_matrix = lda.show_topics(formatted=False, num_words=20)
for i in range(0,len(topics_matrix)):
    print("topic "+str(i)+": ")
    for k in range(0,20):
        print(topics_matrix[i][1][k])
    print('\n')

