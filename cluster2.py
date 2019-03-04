import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


df = pd.read_csv('tweets.csv')
df['text'] = df['text'].astype(str)
df = df.head(5000)

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
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in manual_stop_words])

    # remove special characters
    data['text'] = data['text'].apply(lambda x: [re.sub('[^A-Za-z0-9]+', '', item) for item in x])

    data.columns = ['ID', 'user_screen_name', 'in_reply_to_screen_name', 'text', 'retweeted_screen_name', 'party']

    return data['text']


df_clean = preprocessing(df)
data = []
for index, row in df_clean.iteritems():
    doc = []
    for word in row:
        doc.append(word)
    data.append(doc)

NUM_TOPICS = 10


# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(data)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in data]

# Have a look at how the 20th document looks like: [(word_id, count), ...]
print(corpus[20])

# Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

# Build the LSI model# Build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

print("LDA Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))

print("=" * 20)

print("LSI Model:")

for idx in range(NUM_TOPICS):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lsi_model.print_topic(idx, 10))

print("=" * 20)


