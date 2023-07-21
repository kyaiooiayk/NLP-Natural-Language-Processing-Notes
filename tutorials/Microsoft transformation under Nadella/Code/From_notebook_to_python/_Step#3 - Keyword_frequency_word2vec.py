#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[1]:


"""
What? NLP analysis of some pdf files. This is step#3

The goal is to to analyze Microsoft’s earnings transcripts in pre- and post-Satya Nadella days to extract insights
about how the company’s philosophy and strategy evolved over time. The goal of step#3 is to a word2vec. Word2Vec is 
a statistical method for efficiently learning a standalone word embedding from a text corpus.

Reference: https://mikechoi90.medium.com/investigating-microsofts-transformation-under-satya-nadella-f49083294c35
"""


# # Import libraries

# In[2]:


import eda_utils as eu
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.max_columns', 100)


# ## Scattertext

# In[3]:


corpus_ballmer = pickle.load(open('cleaned_corpus_ball.pickle', 'rb'))


# In[ ]:


corpus_nadella = pickle.load(open('cleaned_corpus_nad.pickle', 'rb'))


# In[ ]:


df_msft_scatter = eu.scatter_msft_df(corpus_ballmer, corpus_nadella)


# In[ ]:


eu.scatterplot(df_msft_scatter)


# ## Changes in Top 10 Words over Time

# In[ ]:


corpus_msft = pickle.load(open('cleaned_corpus_msft.pickle', 'rb'))


# In[ ]:


df_msft = eu.document_term_matrix_df(corpus_msft, CountVectorizer)


# In[ ]:


eu.top_ten_words(corpus_msft, df_msft)


# ## Keyword Frequency (Importance) over Time

# In[ ]:


keyword_list1 = ['cloud', 'ai', 'iot', 'saas', 'license', 'piracy', 'hardware', 'shipment']
eu.keyword_frequency(df_msft, keyword_list1)


# In[ ]:


keyword_list2 = ['commercial', 'consumer', 'margin', 'cost', 'secular', 'differentiate', 'guidance', 'forecast']
eu.keyword_frequency(df_msft, keyword_list2)


# ## Wordcloud

# In[ ]:


eu.wordcloud(corpus_ballmer)


# In[ ]:


eu.wordcloud(corpus_nadella)


# ## Visualizing Word2Vec Word Embeddings using t-SNE

# ### Unigrams

# In[ ]:


eu.word2vec_plot(corpus_msft, 100, 20)


# ### Bigrams

# In[ ]:


corpus_msft_bi = pickle.load(open('cleaned_corpus_bi_msft.pickle', 'rb'))


# In[ ]:


eu.word2vec_plot(corpus_msft_bi, 60, 10)


# In[ ]:




