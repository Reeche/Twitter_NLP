import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
import re
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer


data = pd.read_csv('/Users/richi/Desktop/CSML/NLP/Assignment3/Part_1/tweets.csv')
data['text'] = data['text'].astype(str)
data_copy = data



# get bag of words and remove list in list
def bagofwords(data):
    bag = []
    for tw in range(0, len(data['text'])):
        bag += data['text'][tw]
    return bag


# lower, split, remove stopwords, remove special characters, tokenize
def preprocessing(data):
    """
    Delete stopwords from NLTK and manual list of stopwords

    :param data: pandas data input
    :return: pandas without stopwords
    """
    stemmer = SnowballStemmer("german")

    data['text'] = data['text'].str.lower().str.split()

    #stop_words = stopwords.words('german')

    file = open('german_stopwords.txt', 'r')
    manual_stop_words = file.read()

    # remove stop words
    #data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop_words])
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in manual_stop_words])

    # remove special characters
    data['text'] = data['text'].apply(lambda x: [re.sub('[^A-Za-z0-9]+', '', item) for item in x])

    # stemming
    data['text'] = data['text'].apply(lambda x: [stemmer.stem(item) for item in x])

    # remove empty strings
    data['text'] = data['text'].apply(lambda x: [filter(None, data['text']) for item in x])
    # tokenize
    #tokenizer = RegexpTokenizer(r'\w+')
    #data['text'] = data['text'].apply(lambda x: [tokenizer.tokenize(item) for item in x])


    data.columns = ['ID', 'user_screen_name', 'in_reply_to_screen_name', 'text', 'retweeted_screen_name', 'party']
    data = data.drop(['ID'], axis=1)


    # Uncomment following line to save it as csv
    data.to_csv('cleaned_tweets.csv', header=True, sep=';')
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




preprocessing(data_copy)
