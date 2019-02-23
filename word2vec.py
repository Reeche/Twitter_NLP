import pandas as pd
import nltk
from nltk.corpus import stopwords

data = pd.read_csv('cleaned_tweets.csv', sep=";", header=0)
data['text'] = data['text'].to

#print(data['text'][5000])
# [['buerotweet'], ['vorwaerts'], ['letzte'], ['greichenbach'], ['berlin'], ['treffen'], ['kann']]


bag = []
for tw in range(0, len(data['text'])):
    bag += data['text'][tw]

print(pd.DataFrame(bag))

#bag = data.groupby(['text']).agg(lambda x: x.tolist())