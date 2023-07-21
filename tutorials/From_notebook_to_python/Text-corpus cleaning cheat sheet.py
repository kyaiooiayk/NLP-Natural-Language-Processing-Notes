#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Text-corpus cleaning

It contains a list of useful commands

Reference: various notebooks and the web
"""


# # Import libraries/modules

# In[5]:


import re, string


# # Removing digits in the corpus

# In[2]:


corpus = "I was born on 18th April 1987!"
corpus = re.sub(r'\d+','', corpus)
print(corpus)


# # Lower case the corpus

# In[3]:


corpus = "I was born on 18th April 1987!"
corpus = corpus.lower()
print(corpus)


# # Removing punctuations

# In[6]:


corpus = "I was born on 18th April 1987!"
corpus = corpus.translate(str.maketrans('', '', string.punctuation))
print(corpus)


# # Removing trailing whitespaces

# In[ ]:


"""
Essentially when there are more than one space between words
"""


# In[13]:


corpus = "I was born on 18th  __  __     April 1987!"
corpus = ' '.join([token for token in corpus.split()])
corpus


# In[ ]:





# In[ ]:




