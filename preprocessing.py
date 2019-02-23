import pandas as pd
import nltk
from nltk.corpus import stopwords

data = pd.read_csv('tweets.csv')
data['text'] = data['text'].astype(str)
data_copy = data

# lower, split
data_copy['text'] = data_copy['text'].str.lower().str.split()

stop_words = stopwords.words('german')

file = open('german_stopwords.txt', 'r')
manual_stop_words = file.read()

# remove stop words
data_copy['text'] = data_copy['text'].apply(lambda x: [item for item in x if item not in stop_words])
data_copy['text'] = data_copy['text'].apply(lambda x: [item for item in x if item not in manual_stop_words])

data_copy.columns = ['ID', 'user_screen_name', 'in_reply_to_screen_name', 'text', 'retweeted_screen_name', 'party']
data_copy = data_copy.drop(['ID'], axis=1)

#data_copy.to_csv('nort_nostopwords_tweets.csv', header=True, sep=';')


# count frequencies
all_text = []
for num in range(len(data_copy['text'])):
    all_text += data_copy['text'][num]

fdist1 = nltk.FreqDist(all_text)
#print(fdist1)
print(fdist1.most_common(50))
#fdist1.plot(50, cumulative=True)
