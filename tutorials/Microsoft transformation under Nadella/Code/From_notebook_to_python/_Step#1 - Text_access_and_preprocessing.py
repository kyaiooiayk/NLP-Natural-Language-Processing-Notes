#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[1]:


"""
What? NLP analysis of some pdf files. This is step#1

The goal is to to analyze Microsoft’s earnings transcripts in pre- and post-Satya Nadella days to extract insights
about how the company’s philosophy and strategy evolved over time. The goal of step#1 is to read-in the file
and save a cleaned version of it ready to be used by NLP.

Reference: https://mikechoi90.medium.com/investigating-microsofts-transformation-under-satya-nadella-f49083294c35
"""


# # Import libraries

# In[2]:


#!pip install pyspellchecker
import text_preprocessing_utils as tpu
import pickle
import os


# # Extracting Text from PDF files

# In[ ]:


"""
I retrieved Microsoft’s earnings transcripts in PDFs from Capital IQ API.
    3Q’07–2Q’14: 28 quarters of transcripts in the Steve Ballmer era
    3Q’14–2Q’21: 28 quarters of transcripts in the Satya Nadella era
These are all .pdf files
"""


# In[3]:


directory = r'..//Data/Transcripts'
msft_earnings_dict_orig = {}
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        print("Reading file: " + filename[0:-4])
        msft_earnings_dict_orig[filename[0:-4]] = tpu.text_extractor(os.path.join('../Data/Transcripts', filename))
    else:
        continue


# # Text Preprocessing

# In[ ]:


"""
I then went through a pipeline of text preprocessing steps using NLTK and SpaCy:
    [1] Removed punctuations and numbers
    [2] Removed stopwords
    [3] Lemmatization
    [4] Corrected spelling errors
    [5] Removed people’s names
    
The text are large so it takes some times to process them.
"""


# In[ ]:


# Making a copy so to leave the original untouched
msft_earnings_dict = msft_earnings_dict_orig.copy()


# In[ ]:


# Check the keys
msft_earnings_dict.keys()


# # Remove line breaks, punctuations, and numbers

# In[ ]:


msft_earnings_dict_v2 = tpu.text_preprocessing_pipeline_1(msft_earnings_dict)


# # Tokenization, correct spelling errors, remove stopwords, lemmatization, remove people's names

# In[ ]:


msft_earnings_dict_v3 = tpu.text_preprocessing_pipeline_2(msft_earnings_dict_v2)


# # Remove frequently used words that have no information value

# In[ ]:


msft_earnings_dict_v4 = tpu.remove_custom_stopwords_unigrams(msft_earnings_dict_v3)


# # Bigrams 

# In[ ]:


"""
The key term here is an “n-gram” – a sequence of n words that appear consecutively. One way to create them is to use
TF-IDF. The problem with n-grams is that there are so many potential ones out there. 
"""


# ![image.png](attachment:image.png)

# In[ ]:


msft_earnings_dict_v3_copy = msft_earnings_dict_v3.copy()


# In[ ]:


msft_earnings_dict_bi = tpu.create_bigrams(msft_earnings_dict_v3_copy)


# In[ ]:


msft_earnings_dict_bi_v2 = tpu.remove_custom_stopwords_bigrams(msft_earnings_dict_bi)


# # Pickle the transcript corpus

# In[ ]:


"""
Saving the post-process files
"""


# In[ ]:


with open('cleaned_corpus.pickle', 'wb') as file:
    pickle.dump(msft_earnings_dict_v4, file)


# In[ ]:


with open('cleaned_corpus_bi.pickle', 'wb') as file:
    pickle.dump(msft_earnings_dict_bi_v2, file)


# In[ ]:




