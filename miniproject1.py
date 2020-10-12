#%%
"""
HOUSING DATASET (AMES,LOWA,USA)
DATE : 13/November/2019
SUBMITTED BY:
AYUSH PANWAR   ROLL NO:2013671
GAURAV TRIVEDI ROLL NO:2013685
HARSHIT GUPTA  ROLL NO:2013673
"""
#%%
#import libraries
from sklearn.metrics import accuracy_score, confusion_matrix,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sklearn
import sys
import os
DeprecationWarning('ignore')
os.chdir('D:/machine_learning/boston housing')
DeprecationWarning('ignore')
train = pd.read_csv('train.csv') #creating a dataframe from a csv file.
test = pd.read_csv('test.csv')
#%%
print ("Shape of Train data is:", train.shape)
#1460 rows and 81 columns
print ("Shape of Test data is:", test.shape)
#1459 rows and 80 columns

# %%
train.head()
"""
Data description:
-SalePrice — the property’s sale price in dollars. 
This is the target variable that you’re trying to predict.
-MSSubClass — The building class
-MSZoning — The general zoning classification
-LotFrontage — Linear feet of street connected to property
-LotArea — Lot size in square feet
-Street — Type of road access
-Alley — Type of alley access
-LotShape — General shape of property
-LandContour — Flatness of the property
-Utilities — Type of utilities available
-LotConfig — Lot configuration
And so on.
"""

#%%
#visualize the distribution of the data and check for outliers.
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.show()

#%%
train.SalePrice.describe()
#%%
"""
count=Total no. of rows.
mean=average price of house i.e. $180000(approx.)
std=standard deviation.
min=minimum value.
"""

# %%
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='red')
plt.show()
#%%
import seaborn as sns
sns.distplot(train['SalePrice'].dropna())
plt.show()

#%%
"""
Skew is: 1.8828757597682129
This means that it is highly positive skewed.
"""


# %%
target = np.log(train.SalePrice)
"""
we use np.log() to transform train.SalePrice and
calculate the skewness a second time, as well as 
re-plot the data.
"""
print ("Skew is:", target.skew())
plt.hist(target, color='red')
plt.show()
#%%
sns.distplot(target.dropna())
plt.show()
# %%
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

# %%
corr = numeric_features.corr()
"""
The DataFrame.corr() method displays the correlation
between the columns.
We’ll examine the correlations between the features 
and the target.
"""
print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-10:])
#%%
#plotting heatmap
matrix=train.corr()
f,ax=plt.subplots(figsize=(16,12))
sns.heatmap(matrix,vmax=1,cmap="YlGnBu",linewidth=.5,square=True)
#column- OverallQual is highly correlated
# %%
train.OverallQual.unique()
#We have semi categorical variable OverallQuall
#with a score from 1 to 10
#1 means very poor
#5 means average
#10 means excellent
#%%
train.plot.scatter(x='OverallQual',y='SalePrice')

# %%
quality_pivot = train.pivot_table(index='OverallQual',
values='SalePrice', aggfunc=np.median)
#it creates a spreadsheet style pivot table as a Dataframe.

# %%
quality_pivot

# %%
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# %%
"""
 Use plt.scatter() to generate some scatter plots and visualize 
 the relationship between the Ground LivingArea GrLivArea and 
 SalePrice.
"""
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()

# %%
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

# %%
"""
We will create a new dataframe with some outliers removed.
"""
train = train[train['GarageArea'] < 1200]
train= train[train['GrLivArea']<4000]
# %%
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
#%%
plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))
plt.xlim(-1000,6000) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('GrLive Area')
plt.show()

# %%
#Check for Null values
(train.isnull().sum().sort_values(ascending=False)[:20])


# %%
print ("Unique values are:", train.MiscFeature.unique())
"""
MiscFeature: Miscellaneous feature not covered in other categories

   Elev Elevator
   Gar2 2nd Garage (if not described in garage section)
   Othr Other
   Shed Shed (over 100 SF)
   TenC Tennis Court
   NA   None
These values describe whether or not the house has a shed over
100 sqft,a second garage, and so on.
 """
#%%
#catagorical features
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
# %%
print ("Original: \n")
print (train.Street.value_counts(), "\n")

# %%
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

# %%
print ('Encoded: \n')
print (train.enc_street.value_counts())

# %%
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# %%
def encode(x):
 return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

# %%
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# %%
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# %%
sum(data.isnull().sum() != 0)

# %%
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
# axis=1 column
# %%
#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size=.33)

# %%
from sklearn import linear_model
#Creating object
lr = linear_model.LinearRegression()

# %%
from sklearn.metrics import r2_score
model = lr.fit(X_train, y_train)
y_pred=lr.predict(X)
r2_score(y, y_pred)
# %%
print ("R^2 is: \n", model.score(X_test, y_test))

# %%
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

# %%
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

# %%
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
#%%
#Ridge and Lasso Regression
"""
Ridge and Lasso regression are some of the simple techniques
to reduce model complexity and prevent over-fitting which may 
result from simple linear regression.
 """
from sklearn.linear_model import Ridge
rr=Ridge(alpha=0.01)
rr.fit(X_train,y_train)
test_score=rr.score(X_test,y_test)
print(test_score*100)

# %%
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.0001, max_iter=10e5)
lasso.fit(X_train,y_train)
train_score1=lasso.score(X_train,y_train)
test_score1=lasso.score(X_test,y_test)
print(test_score1*100)

# %%
predictions1 = model.predict(X_test)
from sklearn.metrics import mean_absolute_error
print ('RMAE is: \n', mean_absolute_error(y_test, predictions1))


# %%
