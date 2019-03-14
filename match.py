import pandas as pd
from ast import literal_eval
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('cleaned_tweets_withumlaut.csv', sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
data['scores'] = np.nan

#%%
wertung = pd.read_csv('wertung_noumlaut.csv', sep=",", encoding="ISO-8859-1", low_memory=False)
wertung['word'] = wertung['word'].astype(str)
df1 = wertung[['word', 'score_1']].dropna()
df1.rename(index=str, columns={'score_1': 'scores'}, inplace=True)
df2 = wertung[['word', 'score_2']].dropna()
df2.rename(index=str, columns={'score_2': 'scores'}, inplace=True)
df3 = wertung[['word', 'score_3']].dropna()
df3.rename(index=str, columns={'score_3': 'scores'}, inplace=True)
wertung = df1.append(df2.append(df3, ignore_index=True), ignore_index=True)

#%%
print(data.head(5))
print(wertung.head(5))

for index in data.index:
    word_list = literal_eval(data.get_value(index, 'text'))
    word_matched = []
    for word in word_list:
        for index_ in wertung.index:
            word_ = wertung.get_value(index_, 'word')
            if word.find(word_) != -1:
                score = wertung.get_value(index_, 'scores')
                word_matched.append(score)
    data.iloc[index, 6] = str(word_matched)

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

""" Umlaut replaced with ae, oe, ue
AfD {'1': 4, '2': 43, '3': 30, '4': 2, '5': 40, '6': 2, '7': 30, '8': 3, '9': 11, '10': 9, '11': 68, '12': 52}
CDU/CSU {'1': 189, '2': 609, '3': 612, '4': 13, '5': 382, '6': 54, '7': 400, '8': 80, '9': 421, '10': 100, '11': 1570, '12': 1273}
FDP {'1': 40, '2': 285, '3': 420, '4': 10, '5': 103, '6': 24, '7': 245, '8': 20, '9': 95, '10': 42, '11': 646, '12': 536}
GrÃ¼ne {'1': 280, '2': 707, '3': 1571, '4': 72, '5': 358, '6': 107, '7': 510, '8': 107, '9': 272, '10': 119, '11': 2322, '12': 1848}
Linke {'1': 164, '2': 404, '3': 667, '4': 48, '5': 220, '6': 49, '7': 285, '8': 59, '9': 132, '10': 79, '11': 1078, '12': 939}
SPD {'1': 235, '2': 731, '3': 1548, '4': 132, '5': 471, '6': 53, '7': 490, '8': 104, '9': 404, '10': 168, '11': 2732, '12': 2144}
"""