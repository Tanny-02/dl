"""
Problem 4.d (original code from the shared Notion page, unmodified)

Building a Recommendation System with CBOW Word Embeddings
Problem: Use CBOW word embeddings to build a content-based recommendation system. For
example, recommend similar articles or tweets based on the embeddings of keywords in a
user’s input.
Steps:

1. Data Collection and Preprocessing: Gather a dataset (e.g., a collection of tweets,
articles, or products) with textual content (descriptions, reviews, etc.). Tokenize the text
data, clean the data, and prepare it for training.
2. Train CBOW Model:
a. Train the Word2Vec model using CBOW to learn word embeddings from the
text data or Train the Word2Vec CBOW model on a dataset of articles or
tweets.
b. For a given article (or tweet), calculate the average of the embeddings of all the
words in the document.
c. For each document in the dataset, calculate the cosine similarity between the
document's average word embedding and the user's input embedding.
d. Document Representation: Represent each document (e.g., tweet, product
description, article) as a vector by averaging the embeddings of all words in the
document.
3. Recommendation Logic: For a given input document (e.g., a user’s search or a
product), recommend the most similar documents using cosine similarity between their
vector representations.
"""

# 1. Import necessary packages

import numpy as np
import pandas as pd
import re

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# 2. Load dataset

data = pd.read_csv('twitter_dataset.csv')

# Remove missing tweets
data = data.dropna(subset=['Text'])

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

sentences = data['Text'].apply(preprocess).tolist()


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


# 5. Convert tweets into document vectors

def document_vector(words):

    vectors = [
        model.wv[word]
        for word in words
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)


# Create vectors for all tweets

tweet_vectors = np.array([
    document_vector(words)
    for words in sentences
])


# 6. Take user input

user_input = input("Enter your search query: ")

user_words = preprocess(user_input)

# Convert user input into vector

user_vector = document_vector(user_words)


# 7. Calculate cosine similarity

similarity = cosine_similarity(
    [user_vector],
    tweet_vectors
)[0]


# 8. Recommend top 5 similar tweets

data['Similarity'] = similarity

recommendations = data.sort_values(
    by='Similarity',
    ascending=False
).head(5)


# 9. Display recommendations

print("\nRecommended Tweets:")

for i, tweet in enumerate(recommendations['Text'], 1):

    print("\nRecommendation", i)
    print(tweet)


print("\nRecommendation System Completed!")
