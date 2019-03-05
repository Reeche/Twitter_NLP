import pandas as pd
from ast import literal_eval
import numpy as np

import warnings

##### script for matching tweet.csv with list of words ####


warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('cleaned_tweets.csv', sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
data['scores'] = np.nan

wertung = pd.read_csv('wertung.csv', sep=";", encoding="ISO-8859-1", low_memory=False)
wertung['word'] = wertung['word'].astype(str)

print(data.head(5))
print(wertung.head(5))

for index in data.index:
    word_list = literal_eval(data.get_value(index, 'text'))
    word_matched = []
    for word in word_list:
        for index_ in wertung.index:
            word_ = wertung.get_value(index_, 'word')
            if word.find(word_) != -1:
                score = wertung.get_value(index_, 'score_1')
                word_matched.append(score)
                # score2 = wertung.get_value(index_, 'score_2')
                # if score2 != 'false':
                #     word_matched.append(score2)
                # score3 = wertung.get_value(index_, 'score_3')
                # if score3 != 'false':
                #     word_matched.append(score3)
    data.iloc[index, 7] = str(word_matched)


def f(x):
    return pd.Series(dict(scores="{%s}" % ', '.join(x['scores'])))


party_count = pd.DataFrame(data=data, columns=['party', 'scores']).groupby('party').apply(f)
for index in party_count.index:
    score_dict = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
    }
    scores = party_count.get_value(index, 'scores').replace("[", "").replace("]", "").replace(",", "").replace("{",
                                                                                                               "").replace(
        "}", "").split(' ')
    for score in scores:
        if score != '':
            try:
                number = str(int(float(score)))
                if number in score_dict:
                    score_dict[number] += 1
            except:
                pass
    print(index, score_dict)
