# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:34:41 2023

@author: fkk42
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Download the stopwords dataset
nltk.download('stopwords')

# Load data
data = pd.read_csv('Sentiment.csv')
if data is None:
    print("no data loaded!")

# Keeping only the necessary columns
data = data[['text', 'sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data, test_size=0.1)

# Extracting word features
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# Preprocess and filter stopwords
tweets = []
stopwords_set = set(stopwords.words("english"))
for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if word not in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

# Word features
w_features = get_word_features(get_words_in_tweets(tweets))

# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Counting positive, negative, and neutral tweets in training and test
pos_count_train = len(train[train['sentiment'] == 'Positive'])
neg_count_train = len(train[train['sentiment'] == 'Negative'])
neutral_count_train = len(train[train['sentiment'] == 'Neutral'])

pos_count_test = len(test[test['sentiment'] == 'Positive'])
neg_count_test = len(test[test['sentiment'] == 'Negative'])
neutral_count_test = len(test[test['sentiment'] == 'Neutral'])

# Predictions
pos_cnt = 0
neg_cnt = 0
neutral_cnt = 0
for obj in test['text']:
    res = classifier.classify(extract_features(obj.split()))
    if res == 'Positive':
        pos_cnt += 1
    elif res == 'Negative':
        neg_cnt += 1
    else:
        neutral_cnt += 1

# Display counts and bar charts
print('[Positive] Train/Test: %s/%s ' % (pos_count_train, pos_count_test))
print('[Negative] Train/Test: %s/%s ' % (neg_count_train, neg_count_test))
print('[Neutral] Train/Test: %s/%s ' % (neutral_count_train, neutral_count_test))

# Bar chart for Sentiment Distribution in Training Data
sentiment_counts_train = train['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts_train.plot(kind='bar', color=['red', 'orange', 'blue'])
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Bar chart for Sentiment Distribution in Test Data
sentiment_counts_test = test['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sentiment_counts_test.plot(kind='bar', color=['red', 'orange', 'blue'])
plt.title('Sentiment Distribution in Test Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Bar chart for Predicted Sentiment Distribution
predicted_counts = [neg_cnt,neutral_cnt , pos_cnt]
sentiments = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(8, 6))
plt.bar(sentiments, predicted_counts, color=['red', 'orange', 'blue'])
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
