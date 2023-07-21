#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Part of Speech (POS) taggging

NLTK is a standard python library for natural language processing and computational linguistics.

Reference: https://www.mygreatlearning.com/blog/nltk-tutorial-with-python/?highlight=nlp
"""


# # Import libraries

# In[2]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 


# # Part of Speech (POS) taggging

# In[ ]:


"""
POS tagging is the process of identifying parts of speech of a sentence. It is able to identify nouns, pronouns, 
adjectives etc. in a sentence and assigns a POS token to each word. There are different methods to tag, but we 
will be using the universal style of tagging.
"""


# In[6]:


content = "Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations  that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, and pies."


# In[7]:


content


# In[5]:


sent_tokenize(content)


# In[8]:


word_tokenize(content)


# In[4]:


words= [word_tokenize(i) for i in sent_tokenize(content)] 
pos_tag= [nltk.pos_tag(i, tagset="universal") for i in words] 
for singleTag in pos_tag[0]:
    print(singleTag)


# In[ ]:





# In[ ]:




