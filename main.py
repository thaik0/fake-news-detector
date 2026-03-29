import numpy as np
import pandas as pd # work with tabular data
import itertools
from sklearn.model_selection import train_test_split #split data into training and testing data
from sklearn.feature_extraction.text import TfidfVectorizer # convert text to numerical ML inputs
from sklearn.linear_model import PassiveAggressiveClassifier # classifier model used
from sklearn.metrics import accuracy_score, confusion_matrix # evaluation tools

df = pd.read_csv("news.csv") # read data

print("data shape:")
print(df.shape)

print("\ndata columns:")
print(df.columns)

print("\nfirst 5 rows:")
print(df.head())


labels = df.label # get labels
print(labels.head())

# split the dataset, 80% training 20% learning. random state used for reproducible results.
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

print("\ntrain and test sizes:")
print(len(x_train), len(x_test), len(y_train), len(y_test))