import numpy as np
#%%
import pandas as pd
import nltk
import matplotlib.pyplot as plt
%matplotlib inline
import re
import sys
import os
DeprecationWarning('ignore')
os.chdir('D:/machine_learning/Tweets.csv')
# %%
tweet=pd.read_csv("Tweets.csv")

# %%
tweet.head(n=20)

# %%
tweet.airline.value_counts().plot(kind='pie',autopct='%1.0f%%')

# %%
airline_sentiment=tweet.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')

# %%
features=tweet.iloc[:, 10].values
labels=tweet.iloc[:, 1].values

# %%
processed_features = []

# %%
for sentence in range(0, len(features)):
    
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


# %%
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


print(X_train.shape," ",X_test.shape)

# %%
