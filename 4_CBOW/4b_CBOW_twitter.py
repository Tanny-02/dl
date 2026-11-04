"""
Problem 4.b (code from the shared Notion page)

4B. Implement the Continuous Bag of Words (CBOW) Model for the twitter dataset
(download from Kaggle) using the below steps:
a. Data Collection and Preprocessing: Use Kaggle Twitter dataset or any dataset
with tweets. Tokenize the tweets, clean the text, and prepare it for training.
b. Train Word2Vec CBOW Model
c. Evaluate Results: After training, we’ll check the learned word embeddings and
possibly evaluate using some examples.
"""

# 1. Import necessary packages

import numpy as np
import pandas as pd
import re

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# 2. Load dataset

data = pd.read_csv('twitter_dataset.csv')

print(data.head())


# 3. Preprocess tweets

stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Tokenize text
    words = simple_preprocess(text)

    # Remove stop words
    words = [w for w in words if w not in stop_words]

    return words


# Apply preprocessing

sentences = data['Text'].dropna().apply(preprocess).tolist()

# Remove empty sentences

sentences = [s for s in sentences if len(s) > 0]


# 4. Train Word2Vec using CBOW

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,
    epochs=10
)


# 5. Evaluate word embeddings

print("Vocabulary Size:", len(model.wv))

# Get a word from vocabulary

word = model.wv.index_to_key[0]

print("Selected Word:", word)

# Display word vector

print("Word Vector:")
print(model.wv[word])

# Find similar words

print("Similar Words:")
print(model.wv.most_similar(word, topn=5))


# 6. Calculate similarity between two words

words = model.wv.index_to_key

if len(words) >= 2:

    word1 = words[0]
    word2 = words[1]

    similarity = model.wv.similarity(word1, word2)

    print("Word 1:", word1)
    print("Word 2:", word2)
    print("Similarity:", similarity)


# 7. Save trained model

model.save("twitter_cbow.model")

print("CBOW Model Trained Successfully!")
