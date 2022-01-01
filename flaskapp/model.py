# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:15:53 2021

@author: celine
"""

import pandas as pd
import numpy as np
import re
import timeit
import joblib

import gensim, spacy
import gensim.corpora as corpora
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from skmultilearn.problem_transform import BinaryRelevance
from functions_model import *

# Import dataset
df = pd.read_csv('QueryResults.csv')
df = df[df['Score'] >= 30]
df = df[['Title', 'Body', 'Tags']]
df.reset_index(inplace=True,drop=True)

# Define and clean text
df['Text'] = df['Title'] + ' ' + df['Body']
df['Text'] = df['Text'].apply(clean_text)
df['Text'] = df['Text'].apply(remove_stop_words)
df['Text'] = df['Text'].apply(lemmatize)

# Clean Tags column
df['Tags'] = df['Tags'].apply(clean_tags)

# Find all tags
all_tags = []
for i in range(df.shape[0]):
    all_tags += df.iloc[i, df.columns.get_loc('Tags')]

# Remove tags that are too rare and define new tags
t = Counter(all_tags)
L = []

for k in range(df.shape[0]):
    L.append([elt for elt in df.iloc[k, df.columns.get_loc('Tags')] if (t[elt] >= 60)])
df['new_tags'] = pd.Series(L)

# Remove rows with missing new tags
df = df[df['new_tags'].apply(lambda x: len(x) != 0)]
df.reset_index(inplace=True, drop=True)

# Preprocess text data
X = df.Text.values.tolist()

y = df.new_tags

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


#Use multilabelbinarizer on targets
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(y_train)
test_labels = mlb.transform(y_test)
joblib.dump(mlb, 'mlb.pkl')

# Use tf-idf
vectorizer = TfidfVectorizer(min_df = 0.005, max_df=0.90,sublinear_tf=True)
vectorised_train_documents = vectorizer.fit_transform(X_train).toarray()
vectorised_test_documents = vectorizer.transform(X_test).toarray()
joblib.dump(vectorizer, 'tfidf.pkl')

# Apply PCA
pca = decomposition.PCA(n_components=500)
pca.fit(vectorised_train_documents)
vectorised_train_documents=pca.transform(vectorised_train_documents)
vectorised_test_documents=pca.transform(vectorised_test_documents)
joblib.dump(pca, 'pca.pkl')


#Apply model
svmClassifier = BinaryRelevance(LinearSVC())
svmClassifier.fit(vectorised_train_documents, train_labels)
svmPreds = svmClassifier.predict(vectorised_test_documents)

#save model
joblib.dump(svmClassifier, 'tags_model.pkl')
