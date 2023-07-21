#!/usr/bin/env python
# coding: utf-8

# # Inotroduction

# In[1]:


"""
What? NLP analysis of some pdf files. This is step#2

The goal is to to analyze Microsoft’s earnings transcripts in pre- and post-Satya Nadella days to extract insights
about how the company’s philosophy and strategy evolved over time. the goal of step#2 is 


Reference: https://mikechoi90.medium.com/investigating-microsofts-transformation-under-satya-nadella-f49083294c35
"""


# ## Import Libraries

# In[1]:


import model_utils as mu
import pickle
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# Getting rid of the warning messages
import warnings
warnings.filterwarnings("ignore")


# ## Unigrams

# ### Create Document-Term Matrices

# In[2]:


msft_earnings_dict_cleaned = pickle.load(open('cleaned_corpus.pickle', 'rb'))


# In[ ]:


msft_earnings_dict_ordered = {}

for tscript in sorted(msft_earnings_dict_cleaned):
    msft_earnings_dict_ordered[tscript] = msft_earnings_dict_cleaned[tscript]


# In[ ]:


ballmer_dict = dict(list(msft_earnings_dict_ordered.items())[:len(msft_earnings_dict_ordered)//2]) 
nadella_dict = dict(list(msft_earnings_dict_ordered.items())[len(msft_earnings_dict_ordered)//2:]) 


# In[ ]:


corpus_msft = list(msft_earnings_dict_ordered.values())
corpus_ballmer = list(ballmer_dict.values())
corpus_nadella = list(nadella_dict.values())


# In[ ]:


"""
Split the data in two differetn files for easy access.
"""


# In[ ]:


with open('cleaned_corpus_msft.pickle', 'wb') as f_msft:
    pickle.dump(corpus_msft, f_msft)

with open('cleaned_corpus_ball.pickle', 'wb') as f_ball:
    pickle.dump(corpus_ballmer, f_ball)

with open('cleaned_corpus_nad.pickle', 'wb') as f_nad:
    pickle.dump(corpus_nadella, f_nad)


# #### Count vectorizer

# In[ ]:


cv_msft = mu.document_term_matrix(corpus_msft, CountVectorizer)[0]
cv_ballmer = mu.document_term_matrix(corpus_ballmer, CountVectorizer)[0]
cv_nadella = mu.document_term_matrix(corpus_nadella, CountVectorizer)[0]

X_msft = mu.document_term_matrix(corpus_msft, CountVectorizer)[1]
X_ballmer = mu.document_term_matrix(corpus_ballmer, CountVectorizer)[1]
X_nadella = mu.document_term_matrix(corpus_nadella, CountVectorizer)[1]


# In[ ]:


word_freq_msft = mu.word_frequency(cv_msft, X_msft)
word_freq_ballmer = mu.word_frequency(cv_ballmer, X_ballmer)
word_freq_nadella = mu.word_frequency(cv_ballmer, X_ballmer)


# #### Tf-idf vectorizer

# In[ ]:


cv_tfidf_msft = mu.document_term_matrix(corpus_msft, TfidfVectorizer)[0]
cv_tfidf_ballmer = mu.document_term_matrix(corpus_ballmer, TfidfVectorizer)[0]
cv_tfidf_nadella = mu.document_term_matrix(corpus_nadella, TfidfVectorizer)[0]

X_tfidf_msft = mu.document_term_matrix(corpus_msft, TfidfVectorizer)[1]
X_tfidf_ballmer = mu.document_term_matrix(corpus_ballmer, TfidfVectorizer)[1]
X_tfidf_nadella = mu.document_term_matrix(corpus_nadella, TfidfVectorizer)[1]


# ### Topic Modeling

# #### Non-negative matrix factorization (NMF), Count vectorizer

# In[ ]:


X_msft.shape


# In[ ]:


X_ballmer.shape


# In[ ]:


X_nadella.shape


# In[ ]:


top_topic_words_nmf_msft = mu.topic_model(cv_msft, X_msft, NMF, 6, 6)[0]
top_topic_words_nmf_ballmer = mu.topic_model(cv_ballmer, X_ballmer, NMF, 6, 6)[0]
top_topic_words_nmf_nadella = mu.topic_model(cv_nadella, X_nadella, NMF, 6, 6)[0]

top_topic_words_nmf_msft


# In[ ]:


topic_clustering_nmf_msft = mu.topic_model(cv_msft, X_msft, NMF, 6, 6)[1].argmax(axis=1)
topic_clustering_nmf_ballmer = mu.topic_model(cv_ballmer, X_ballmer, NMF, 6, 6)[1].argmax(axis=1)
topic_clustering_nmf_nadella = mu.topic_model(cv_nadella, X_nadella, NMF, 6, 6)[1].argmax(axis=1)

topic_clustering_nmf_msft


# #### Latent Dirichlet Allocation (LDA) - Count vectorizer

# In[ ]:


top_topic_words_lda_msft = mu.topic_model(cv_msft, X_msft, LatentDirichletAllocation, 6, 6)[0]
top_topic_words_lda_ballmer = mu.topic_model(cv_ballmer, X_ballmer, LatentDirichletAllocation, 6, 6)[0]
top_topic_words_lda_nadella = mu.topic_model(cv_nadella, X_nadella, LatentDirichletAllocation, 6, 6)[0]

top_topic_words_lda_msft


# In[ ]:


topic_clustering_lda_msft = mu.topic_model(cv_msft, X_msft, LatentDirichletAllocation, 6, 6)[1].argmax(axis=1)
topic_clustering_lda_ballmer = mu.topic_model(cv_ballmer, X_ballmer, LatentDirichletAllocation, 6, 6)[1].argmax(axis=1)
topic_clustering_lda_nadella = mu.topic_model(cv_nadella, X_nadella, LatentDirichletAllocation, 6, 6)[1].argmax(axis=1)

topic_clustering_lda_msft


# #### Non-negative matrix factorization (NMF), Tf-idf vectorizer

# In[ ]:


top_topic_words_nmf_tfidf_msft = mu.topic_model(cv_tfidf_msft, X_tfidf_msft, NMF, 6, 6)[0]
top_topic_words_nmf_tfidf_ballmer = mu.topic_model(cv_tfidf_ballmer, X_tfidf_ballmer, NMF, 6, 6)[0]
top_topic_words_nmf_tfidf_nadella = mu.topic_model(cv_tfidf_nadella, X_tfidf_nadella, NMF, 6, 6)[0]

top_topic_words_nmf_tfidf_msft


# In[ ]:


topic_clustering_nmf_tfidf_msft = mu.topic_model(cv_tfidf_msft, X_tfidf_msft, NMF, 6, 6)[1].argmax(axis=1)
topic_clustering_nmf_tfidf_ballmer = mu.topic_model(cv_tfidf_ballmer, X_tfidf_ballmer, NMF, 6, 6)[1].argmax(axis=1)
topic_clustering_nmf_tfidf_nadella = mu.topic_model(cv_tfidf_nadella, X_tfidf_nadella, NMF, 6, 6)[1].argmax(axis=1)

topic_clustering_nmf_tfidf_msft


# ### Clustering

# In[ ]:


mu.kmeans_clustering(X_tfidf_msft, 3)


# In[ ]:


# silhouette coefficient and SSE are two metrics used to 
mu.silhouette_coeff_sse(X_tfidf_msft)


# In[ ]:




