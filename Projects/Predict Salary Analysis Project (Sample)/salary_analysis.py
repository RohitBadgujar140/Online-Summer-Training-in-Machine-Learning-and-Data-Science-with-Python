#!/usr/bin/env python
# coding: utf-8

# # Salary Prediction Analysis (Sample Project)

# ML Model Deployment with Flask on Heroku

# # Problem Defination

# A sample data set describing candidate Experience and offered pay based on interview, test score. Using this data predict salary of the candidate.
# 
# Experience : Total number of Experience of candidate
# 
# Interview Score : Marks given to candidate
# 
# test score : Marks earned by giving test.

# # Data Gathering

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[2]:


# reading dataset
df = pd.read_csv("hiring.csv")
df


# # Data Preparation

# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.corr()


# In[6]:


df.describe()


# # Feature Engineering

# In[7]:


# fill score with mean value

df['test_score'].fillna(df['test_score'].mean() , inplace=True)


# In[8]:


df


# In[9]:


df['experience'].fillna(0, inplace=True)


# In[10]:


df


# In[11]:


def string_to_number(word):
    dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6 , 'seven':7, 'eight':8, 'nine':9,
            'ten':10, 'eleven':11, 'twelve':12, 0:0}
    return dict[word]


# In[12]:


df['experience'] = df['experience'].apply(lambda x: string_to_number(x))
df


# # Exploratory Data Analysis

# In[13]:


x = df['experience']
y = df['salary']
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(x,y,color='red',marker='*')
plt.show()


# In[14]:


# Classify Features as x and Label as y
x = df.iloc[:,:3]
y = df.iloc[:,-1]


# In[15]:


# Classify Dataset into Training and Testing
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.1, random_state=1)


# # Modelling

# In[16]:


# creating linear regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[17]:


# train model by providing training data
Mymodel = model.fit(xtrain,ytrain)


# # Model Evaluation

# In[18]:


# Prediction
y_pred = Mymodel.predict(xtest)
y_pred


# In[19]:


comparision = pd.DataFrame(y_pred, ytest)
comparision


# In[20]:


# Manual Prediction
y_manual = Mymodel.predict([[5,8,7]])
y_manual


# # Accuracy Score

# In[22]:


# 1) Mean Squared Error

from sklearn.metrics import mean_squared_error

# calculate errors
errors = mean_squared_error(ytest, y_pred)
# report error
print(errors)


# In[23]:


# 2) Root Mean Squared Error

from sklearn.metrics import mean_squared_error

# calculate errors
errors = mean_squared_error(ytest, y_pred, squared=False)
# report error
print(errors)


# In[24]:


# 3) Mean Absolute Error

from sklearn.metrics import mean_absolute_error

# calculate errors
errors = mean_absolute_error(ytest, y_pred)
# report error
print(errors)


# In[ ]:


# save the model using ickle
import pickle
pickle.dump(Mymodel, open("salary_analysis.pkl", "wb"))


# In[ ]:


model = pickle.load(open('salary_analysis.pkl', 'rb'))


# In[ ]:


model.predict([[5,8,7]])

