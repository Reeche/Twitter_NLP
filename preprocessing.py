import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

    stop_words = stopwords.words('german')

    file = open('german_stopwords.txt', 'r')
    manual_stop_words = file.read()

    # remove stop words
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
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

data_clean = preprocessing(data)


# count frequencies
#all_text = []
#for num in range(len(data_clean['text'])):
#    all_text += data_clean['text'][num]

#fdist1 = nltk.FreqDist(all_text)
#print(fdist1.most_common(50))
#fdist1.plot(50, cumulative=True)



# get bag of words
bag = []
for tw in range(0, len(data_clean['text'])):
    bag += data['text'][tw]

#word2vec model
model = Word2Vec(bag, min_count=100)
words = list(model.wv.vocab)


X = model[model.wv.vocab]

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(X)
print(tsne_results.shape)

plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
for i, word in enumerate(words):
	plt.annotate(word, xy=(tsne_results[i, 0], tsne_results[i, 1]))
plt.show()








