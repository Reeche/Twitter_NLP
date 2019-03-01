import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.cluster import KMeansClusterer
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics
import numpy as np

data = pd.read_csv('tweets.csv')
data['text'] = data['text'].astype(str)
data_copy = data

# lower, split
def preprocessing(data):
    """
    Delete stopwords from NLTK and manual list of stopwords

    :param data: pandas data input
    :return: pandas without stopwords
    """
    data['text'] = data['text'].str.lower().str.split()

    #stop_words = stopwords.words('german')

    file = open('german_stopwords.txt', 'r')
    manual_stop_words = file.read()

    # remove stop words
    #data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in manual_stop_words])

    # remove special characters
    data['text'] = data['text'].apply(lambda x: [re.sub('[^A-Za-z0-9]+', '', item) for item in x])

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    data['text'] = data['text'].apply(lambda x: [tokenizer.tokenize(item) for item in x])


    data.columns = ['ID', 'user_screen_name', 'in_reply_to_screen_name', 'text', 'retweeted_screen_name', 'party']
    data = data.drop(['ID'], axis=1)

    # Uncomment following line to save it as csv
    #data_copy['text'].to_csv('cleaned_tweets_only.csv', header=True, sep=';')
    return data


def count_freq():
    all_text = []
    for num in range(len(data_clean['text'])):
       all_text += data_clean['text'][num]

    fdist1 = nltk.FreqDist(all_text)
    print(fdist1.most_common(50))
    fdist1.plot(50, cumulative=True)
    plt.show()
    pass


# get bag of words
def bagofwords(data):
    bag = []
    for tw in range(0, len(data['text'])):
        bag += data['text'][tw]
    return bag

def clustering(data, number):

    kmeans = cluster.KMeans(n_clusters=number, init='k-means++', max_iter=100, n_init=1)
    assigned_clusters = kmeans.fit(data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # assess goodness
    silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')

    return assigned_clusters, kmeans.score(X), silhouette_score


data_clean = preprocessing(data)
bagofwords(data_clean)

model = Word2Vec(bagofwords(data), min_count=100)
words = list(model.wv.vocab)

X = model[model.wv.vocab]


for n in range(2, 3):
    for p in range(5, 50, 5):
        for l in range(10, 100, 10):
            tsne = TSNE(n_components=n, perplexity=p, learning_rate=l)
            tsne_results = tsne.fit_transform(X)
            for k in range(10, 20):
                assigned_clusters, score, silhouette_score = clustering(X, k)
                print(" number of components", n, "perplexity", p, "cluster", k, "learning rate", l, "score", score, "silhouette", silhouette_score)

#  number of components 2 perplexity 10 cluster 15 learning rate 90 score -0.3707258 silhouette 0.0008933469

#plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=assigned_clusters.labels_,)
#for i, word in enumerate(words):
#    plt.annotate(word, xy=(tsne_results[i, 0], tsne_results[i, 1]))
#plt.show()



def assessquality(NUM_CLUSTERS, data):
    #for i, word in enumerate(words):
    #   print(word + ":" + str(assigned_clusters[i]))

    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    #print("Cluster id labels for inputted data")
    #print(labels)
    #print("Centroids data")
    #print(centroids)

    #print("Score (Opposite of the value of data on the K-means objective which is Sum of distances of samples to their closest cluster center):")
    #print(kmeans.score(X))

    silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')

    #print("Silhouette_score: ")
    #print(silhouette_score)
    return kmeans.score(X), silhouette_score

