#!/usr/bin/env python
# coding: utf-8

# # EDA on Times Series Dataset

# In[2]:


import pandas as pd

data = pd.read_csv('data.csv')
location = data.iloc[:, 17:] # only location column considered
location[['time']] = data[['time']]
# print(used.columns)
row_diffs = location.set_index('time').diff() 
#location.head()


# In[3]:


row_diffs = row_diffs[row_diffs.columns].apply(abs) # convert all difference between rows to abs for sum up


# In[5]:


row_diffs['diff_sum'] = row_diffs[list(row_diffs.columns)].sum(axis=1)


# In[6]:


row_diffs.plot(y='diff_sum', style='o', use_index=True, figsize=(20,5))


# In[7]:


row_diffs.nlargest(10, 'diff_sum')['diff_sum'] # among all


# In[24]:


first_30min = row_diffs[:120]
first_30min.nlargest(10, 'diff_sum')['diff_sum']


# In[1]:


last_40min = row_diffs[-120:] # last 20min
last_40min.nlargest(10, 'diff_sum')['diff_sum']


# In[39]:


last_20min = row_diffs[-120:] # last 20min
last_20min.nsmallest(10, 'diff_sum')['diff_sum']


# In[34]:


last_20min = row_diffs[-80:] # last 20min
last_20min.nlargest(10, 'diff_sum')['diff_sum']


# In[64]:


row_diffs[:10]


# In[6]:


state = [0]*40
state.append(20)
import numpy as np
np.array(state)


# In[41]:


data[40:-72] 

