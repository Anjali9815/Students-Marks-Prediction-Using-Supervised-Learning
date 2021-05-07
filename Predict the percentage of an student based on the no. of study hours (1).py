#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model, preprocessing


# In[2]:


data = pd.read_csv('Data.csv')


# In[3]:


data


# In[4]:


x=data.No_of_Hours_students_studies.to_numpy()
y=data.Marks.to_numpy()


# In[5]:


plt.scatter(x,y)


# In[6]:


reg=linear_model.LinearRegression()
reg.fit(data[['No_of_Hours_students_studies']],data.Marks)


# In[7]:


mark=reg.predict([[2]])
np.round(mark)


# In[ ]:




