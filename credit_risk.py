#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
import warnings
import time
import sys
import os 

DeprecationWarning('ignore')
warnings.filterwarnings('ignore',message="don't have warning")
os.chdir('D:/machine_learning/DATA_SET/credit_risk')
#%%
from sklearn.tree import DecisionTreeClassifier

#%%
tf=pd.read_csv('credit_risk.csv')

#%%
tf.head()

#%%
tf.sample(12)

#%%
tf.describe()

#%%
tf.isnull().sum()

#%%
tf.Gender[tf.Gender == 'Male'] = 1
tf.Gender[tf.Gender == 'Female'] = 2
tf.Married[tf.Married == 'Yes'] = 1
tf.Married[tf.Married == 'No'] = 2
tf.Education[tf.Education == 'Graduate'] = 1
tf.Education[tf.Education == 'Not Graduate'] = 2
tf.Self_Employed[tf.Self_Employed == 'Yes'] = 1
tf.Self_Employed[tf.Self_Employed == 'No'] = 2
tf.Property_Area[tf.Property_Area == 'Rural'] = 1
tf.Property_Area[tf.Property_Area == 'Urban'] = 2
tf.Property_Area[tf.Property_Area == 'Semiurban']= 3
tf.Dependents[tf.Dependents=='3+']=3
#%%
tf.head()
#%%
train,test = train_test_split(tf, test_size=0.2, random_state=12)

#%%
clf = DecisionTreeClassifier()

#%%
train.shape

#%%
test.shape

#%%
train.isnull().sum()
#%%
def fill_Gender(tf):
    median=  1
    tf['Gender'].fillna(median, inplace = True)
    return tf

def fill_Married(tf):
    median=  1
    tf['Married'].fillna(median, inplace = True)
    return tf
def fill_Dependents(tf):
    median=  0
    tf['Dependents'].fillna(median, inplace = True)
    return tf
def fill_Self_Employed(tf):
    median=  2
    tf['Self_Employed'].fillna(median, inplace = True)
    return tf
def fill_LoanAmount(tf):
    mean=  142.5717
    tf['LoanAmount'].fillna(mean, inplace = True)
    return tf
def fill_Loan_Amoount_Term(tf):
    median=  360
    tf['Loan_Amount_Term'].fillna(median, inplace = True)
    return tf
def fill_Credit_Historys(tf):
    median=  1
    tf['Credit_History'].fillna(median, inplace = True)
    return tf

def encode_feature(tf):
    tf = fill_Gender(tf)
    tf=fill_Married(tf)
    tf=fill_Dependents(tf)
    tf=fill_Self_Employed(tf)
    tf=fill_LoanAmount(tf)
    tf=fill_Loan_Amoount_Term(tf)
    tf=fill_Credit_Historys(tf)
    return(tf)
#%%
tf=encode_feature(tf)
#%%

train = encode_feature(train)
test = encode_feature(test)


#%%
def x_and_y(tf):
    x = tf.drop(["Loan_Status","Loan_ID","ApplicantIncome","Loan_Amount_Term","Gender","Education","Dependents","Property_Area","Self_Employed"],axis=1)
    y = tf["Loan_Status"]
    return x,y
x_train,y_train = x_and_y(train)
x_test,y_test = x_and_y(test)
#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

#%%
log_model = DecisionTreeClassifier(criterion='entropy')
log_model.fit(x_train,y_train)
prediction = log_model.predict(x_train)
score = accuracy_score(y_train,prediction)
print(score)

#%%
y_train.shape

#%%
x_train.columns

#%%
log_model = DecisionTreeClassifier(criterion='entropy')
log_model.fit(x_train,y_train)
prediction = log_model.predict(x_test)
score1 = accuracy_score(y_test,prediction)
print(score1)

#%%


