import pandas as pd
from ast import literal_eval
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# data = pd.read_csv('data_lst_processed.csv', sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
# data['scores'] = np.nan

data = pd.read_csv('cleaned_tweets.csv', sep=";", header=0, encoding="ISO-8859-1", low_memory=False)
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
