"""
Problem 4.c (original code from the shared Notion page, unmodified)

Implement the Continuous Bag of Words (CBOW) Model for the given  textual
document 3 using the below steps:
a. Data preparation
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


# 2. Load textual document

file = open('document3.txt', 'r', encoding='utf-8')
text = file.read()
file.close()


# 3. Preprocess text

# Convert text into lowercase
text = text.lower()

# Split text into sentences
sentences = re.split(r'[.!?]+', text)

# Tokenize sentences
stop_words = set(ENGLISH_STOP_WORDS)

data = []

for sentence in sentences:

    words = simple_preprocess(sentence)

    # Remove stop words
    words = [w for w in words if w not in stop_words]

    if len(words) > 0:
        data.append(words)


print("Preprocessed Data:")
print(data)


# 4. Train Word2Vec using CBOW

model = Word2Vec(
    sentences=data,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=100
)


# 5. Evaluate word embeddings

print("Vocabulary Size:", len(model.wv))

# Display word vector

word = 'influenza'

print("Word Vector:")
print(model.wv[word])


# 6. Find similar words

print("Similar Words:")
print(model.wv.most_similar(word, topn=5))


# 7. Calculate similarity between two words

similarity = model.wv.similarity('influenza', 'virus')

print("Similarity:", similarity)


# 8. Save trained model

model.save("document3_cbow.model")

print("CBOW Model Trained Successfully!")
