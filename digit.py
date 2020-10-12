#%%
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import warnings
import time
import sys
import os 

DeprecationWarning('ignore')
warnings.filterwarnings('ignore',message="don't have warning")
os.chdir('D:/machine_learning/digit-recognizer')
#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#%%
df=pd.read_csv('train.csv')

#%%
df.head()

#%%
image=df.iloc[0:1,1:]
#%%
image
#%%
plt.imshow(image.values.reshape(28,28))

#%%
image=df.iloc[3:4,1:]
#%%
image

#%%
plt.imshow(image.values.reshape(28,28))

#%%
train, test = train_test_split(df,test_size=0.2,random_state = 12)


#%%
def x_and_y(df):
    x = df.drop(["label"],axis=1)
    y = df["label"]
    return x,y
x_train,y_train = x_and_y(train)
x_test,y_test = x_and_y(test)

#%%
from sklearn.tree import DecisionTreeClassifier

#%%
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_test)
score1=accuracy_score(y_test,y_predict)
print(score1)

#%%
rfc=RandomForestClassifier(n_estimators=20,criterion='entropy')
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_train)
score2=accuracy_score(y_train,y_predict)
print(score2)

#%%
