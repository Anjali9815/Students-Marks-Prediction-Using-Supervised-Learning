#!/usr/bin/env python
# coding: utf-8

# In[37]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model, preprocessing


# In[31]:


data = pd.read_csv('Data.csv')


# In[32]:


data


# In[33]:


x=data.No_of_Hours_students_studies.to_numpy()
y=data.Marks.to_numpy()


# In[34]:


plt.scatter(x,y)


# In[35]:


reg=linear_model.LinearRegression()
reg.fit(data[['No_of_Hours_students_studies']],data.Marks)


# In[36]:


mark=reg.predict([[2]])
np.round(mark)


# In[ ]:




