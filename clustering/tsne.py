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
from sklearn.feature_extraction.text import TfidfVectorizer

def clustering(data, number):
    """
    This function does the clustering and returns some measures for the goodness of fit

    :param data: input data
    :param number: number of clusters
    :return: assigned clusters, kmeans score, silhouette score
    """

    kmeans = cluster.KMeans(n_clusters=number, init='k-means++', max_iter=100, n_init=1)
    assigned_clusters = kmeans.fit(data)


    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for idx, label in enumerate(labels):
        clustering[label].append(idx)

    # assess goodness
    silhouette_score = metrics.silhouette_score(data, labels, metric='euclidean')

    return assigned_clusters, kmeans.score(X), silhouette_score


data_clean = preprocessing(data)
bagofwords(data_clean)

model = Word2Vec(bagofwords(data), min_count=100)
words = list(model.wv.vocab)

X = model[model.wv.vocab]

def grid_search():
    results = []
    for run in range(0, 5):
        for n in range(2, 4):
            for p in range(5, 50, 5):
                for l in range(10, 1000, 50):
                    tsne = TSNE(n_components=n, perplexity=p, learning_rate=l)
                    tsne_results = tsne.fit_transform(X)
                    for k in range(5, 200, 10):
                        assigned_clusters, score, silhouette_score = clustering(X, k)
                        #print(" number of components", n, "perplexity", p, "cluster", k, "learning rate", l, "score", score, "silhouette", silhouette_score)
                        results.append([run, n, p, k, l, score, silhouette_score])
                print(results)
    results_pd = pd.DataFrame(results)
    results_pd.columns = ['run', 'no_components', 'perplexity', 'cluster', 'learning', 'score', 'silhouette']
    print(results_pd)
    #results_pd.to_csv('grid_results.csv', header=True, sep=';')
    pass




# plot
# according to score: 2             45         5       660 looks better
# according to sihl:  3             30         85      660
tsne = TSNE(n_components=2, perplexity=45, learning_rate=660)
tsne_results = tsne.fit_transform(X)
assigned_clusters, score, silhouette_score = clustering(X, 5)



plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=assigned_clusters.labels_,)
for i, word in enumerate(words):
   plt.annotate(word, xy=(tsne_results[i, 0], tsne_results[i, 1]))
plt.show()



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

