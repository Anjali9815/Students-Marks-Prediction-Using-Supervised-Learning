#!/usr/bin/env python
# coding: utf-8

# # Spark Foundation Data science & Buisness Analyst Task:01
#     Name: Anjali Jha
#     

# # Prediction using Supervised ML
# Predict the percentage of an student based on the no. of study hours.

# # Step I: Importing all the relevant libraries 

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model


# # Step II: Importing dataset

# In[4]:


data = pd.read_csv('Data.csv')
data


# # Step III: Visualizing Data

# In[5]:


x=data.No_of_Hours_students_studies.to_numpy()
y=data.Marks.to_numpy()
plt.scatter(x,y,color='green')
plt.xlabel('Student studies (hours)')
plt.ylabel('Marks')


# # Step IV: Training Model (Linear Regresssion Algorithm)

# In[6]:


reg=linear_model.LinearRegression()
reg.fit(data[['No_of_Hours_students_studies']],data.Marks)


# # Step V: Prediction of Marks 

# In[7]:


reg.predict([[2]])


# # LInear Regresssion
# We have equation as y=mx+c
# where,   
# m is the slope/coefficent,
# b is the intercept,
# x is independent value,
# y is dependent value, 
# 
# so we have,
# Marks = m*Sudent studies(Hours) + c

# In[8]:


reg.coef_


# In[9]:


reg.intercept_


# #Marks = m*Sudent studies(Hours) + c
# y= 9.77580339*2+2.48367340537321
# y

# In[12]:


marks1=9.77580339*2+2.48367340537321
marks1


# In[ ]:




