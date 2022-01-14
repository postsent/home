#!/usr/bin/env python
# coding: utf-8

# # Numpy

# # TODO

# - list to numpy
# - optional param
#     - axis = 0: col, 1 is row, since 1 is updown

# In[12]:


from IPython.core.interactiveshell import InteractiveShell
get_ipython().ast_node_interactivity = 'all'


# # Basic

# ## matrix

# In[1]:


import numpy as np
mat = [
    [1,2,3],
    [4,5,6],
]
mat = np.array(mat)
mat


# ## vector

# In[15]:


v1_list = [1,2,3]
v1 = np.array(v1_list)
v2_list = [3,4,5]
v2 = np.array(v2_list)
v1
v2


# ### concatenate

# In[ ]:





# ## optional params

# ### axis

# In[58]:


mat
mat.sum(axis=0) # 0, 1 - row and col
mat.sum(axis=1)


# ## Math operation

# ### Square

# In[29]:


v1**2 == np.square(v1)


# ### Dot product

# In[24]:


# The math interpretation
res = 0
for e1, e2 in zip(v1_list, v2_list):
    res += e1 * e2
# dot product, numpy
v1 @ v2 == np.dot(v1, v2) == v1.dot(v2) == res
res


# ### ------Linear algebra-----

# ### norm

# Frobenius Norm
# 
# The Frobenius norm is the generalization to $R^2$ of the already known norm function for vectors 
# 
# $$\| \vec a \| = \sqrt {{\vec a} \cdot {\vec a}} $$
# 
# For a given $R^2$ matrix A, the frobenius norm is defined as:
# 
# $$\|\mathrm{A}\|_{F} \equiv \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n}\left|a_{i j}\right|^{2}}$$
# 

# In[60]:


np.linalg.norm(mat) == np.sqrt(np.sum(mat**2))


# ## Other

# ### Get first n column

# In[9]:


mat[:, 1] # Get only that particular column
mat[:, 0:1] # Get first n cols

