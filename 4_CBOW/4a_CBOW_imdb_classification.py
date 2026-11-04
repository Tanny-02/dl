"""
Problem 4.a (code from the shared Notion page)

Implement the CBOW for document classification using deep learning.

Steps (for Implementing CBOW for Document Classification):

1. Data Preparation: You need a labeled dataset for document classification. You
can use datasets like the IMDb movie reviews dataset (for sentiment
classification) or 20 Newsgroups dataset (for topic classification).
Load and Preprocess the data:
• from tensorflow.keras.datasets import imdb # For loading
• Pad sequences to ensure uniform input length
• Convert the IMDB dataset to a list of sentences (each sentence is a list of word)
• Tokenize the text into words.
• Lowercase the text and remove stop words.
• Create a vocabulary from the training data.
2. Define CBOW Model: CBOW learns to predict a target word based on
surrounding context words. For document classification, you need to modify the
CBOW model to work with document-level contexts. In CBOW, the context
consists of words surrounding the target word, but for documents, you will use the
embeddings from all the words in the document.
3. Model Architecture:
• Embedding Layer: First, use the CBOW architecture to learn word
embeddings from the text corpus.
• Document Representation: After learning the word embeddings, represent a
document by averaging the embeddings of all words in the document.
• Fully Connected Layer: Use this document representation as input to a neural
network (e.g., a dense layer) for classification.
• Train word2vec model with CBOW (sg=0 for CBOW)
• Convert words to embeddings
• Build the feed forward neural network for document classification
4. Evaluate the Model on test set using loss and accuracy.


IMDB version code
"""

# 1. Import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from tensorflow import keras
from tensorflow.keras import layers

import re


# 2. Load dataset

data = pd.read_csv('IMDB Dataset.csv')

# Convert sentiment into numbers
data['sentiment'] = data['sentiment'].map({
    'positive': 1,
    'negative': 0
})


# 3. Preprocess text

stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = re.sub('<.*?>', ' ', text)
    words = simple_preprocess(text)
    words = [w for w in words if w not in stop_words]
    return words

data['words'] = data['review'].apply(preprocess)


# 4. Separate input and output

X = data['words'].tolist()
y = data['sentiment'].values

# Split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Train Word2Vec using CBOW

word2vec = Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,
    epochs=5
)


# 6. Convert documents into embeddings

def document_vector(words):
    vectors = [
        word2vec.wv[word]
        for word in words
        if word in word2vec.wv
    ]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)


X_train_vec = np.array([document_vector(w) for w in X_train])
X_test_vec = np.array([document_vector(w) for w in X_test])


# 7. Define Feed Forward Neural Network

model = keras.Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# 8. Compile model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# 9. Train model

history = model.fit(
    X_train_vec, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# 10. Evaluate model

loss, accuracy = model.evaluate(X_test_vec, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# 11. Plot training loss and accuracy

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
