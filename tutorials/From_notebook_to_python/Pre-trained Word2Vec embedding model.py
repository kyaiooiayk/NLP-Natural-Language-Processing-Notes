#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# **What?** Pre-trained Word2Vec embedding model
# 
# <br></font>
# </div>

# # Import modules

# In[25]:


import os
import wget
import gzip
import shutil
#This module ignores the various types of warnings generated
import warnings 
warnings.filterwarnings("ignore") 
#This module helps in retrieving information on running processes and system resource utilization
import psutil 
from psutil import virtual_memory
import time 
from gensim.models import Word2Vec, KeyedVectors
import spacy


# # Import pre-trained model

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Let us take an example of a pre-trained word2vec model, and how we can use it to look for most similar words. 
# - We will use the Google News vectors embeddings. https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
# - **ATTENTION!** the file sizr is: 1.65GB it will take a while to download. The decompressed size is over 3GB
# 
# <br></font>
# </div>

# In[2]:


gn_vec_path = "GoogleNews-vectors-negative300.bin"
if not os.path.exists("GoogleNews-vectors-negative300.bin"):
    if not os.path.exists("./GoogleNews-vectors-negative300.bin"):
        # Downloading the reqired model
        if not os.path.exists("./GoogleNews-vectors-negative300.bin.gz"):
            if not os.path.exists("GoogleNews-vectors-negative300.bin.gz"):
                wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
            gn_vec_zip_path = "GoogleNews-vectors-negative300.bin.gz"
        else:
            gn_vec_zip_path = "./GoogleNews-vectors-negative300.bin.gz"
        # Extracting the required model
        with gzip.open(gn_vec_zip_path, 'rb') as f_in:
            with open(gn_vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        gn_vec_path = "./" + gn_vec_path

print(f"Model at {gn_vec_path}")


# In[4]:


process = psutil.Process(os.getpid())
mem = virtual_memory()


# In[10]:


pretrainedpath = gn_vec_path

# Load W2V model. This will take some time, but it is a one time effort! 
pre = process.memory_info().rss
# Check memory usage before loading the model
print("Memory used in GB before Loading the Model: %0.2f"%float(pre/(10**9))) 
print('-'*10)

# Start the timer
start_time = time.time() 
# Toal memory available
ttl = mem.total 

# Load the model
w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True) 
# Calculate the total time elapsed since starting the timer
print("%0.2f seconds taken to load"%float(time.time() - start_time)) 
print('-'*10)

print('Finished loading Word2Vec')
print('-'*10)

post = process.memory_info().rss
# Calculate the memory used after loading the model
print("Memory used in GB after Loading the Model: {:.2f}".format(float(post/(10**9)))) 
print('-'*10)

# Percentage increase in memory after loading the model
print("Percentage increase in memory usage: {:.2f}% ".format(float((post/pre)*100))) 
print('-'*10)

# Number of words in the vocabulary. 
print("Numver of words in vocablulary [Mil]: " + str(len(w2v_model.key_to_index)/1.e6)) 


# In[ ]:


"""
How many things can we do?
we can inspect the methods with dir
"""


# In[13]:


dir(w2v_model)


# In[15]:


# Let us examine the model by knowing what the most similar words are, for a given word!
w2v_model.most_similar('beautiful')


# In[17]:


# Let us try with another word! 
w2v_model.most_similar('rome')


# In[18]:


# What is the vector representation for a word? 
w2v_model['computer']


# In[ ]:


# What if I am looking for a word that is not in this vocabulary?
w2v_model['practicalnlp']


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - Two things to note while using pre-trained models: 
# - [1] Tokens/Words are always lowercased. If a word is not in the vocabulary,   the model throws an exception.
# - [2] So, it is always a good idea to encapsulate those statements in try/except blocks.
# 
# <br></font>
# </div>

# # Getting the embedding representation for full text

# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - We have seen how to get embedding vectors for single words. 
# - How do we use them to get such a representation for a full text? 
# - A simple way is to just sum or average the embeddings for individual words. 
# - Let us see a small example using another NLP library Spacy
# 
# <br></font>
# </div>

# In[23]:


get_ipython().run_line_magic('time', "nlp = spacy.load('en_core_web_sm')")
# process a sentence using the model
mydoc = nlp("Canada is a large country")
#Get a vector for individual words
#print(doc[0].vector) #vector for 'Canada', the first word in the text 
print(mydoc.vector) #Averaged vector for the entire sentence


# <div class="alert alert-block alert-info">
# <font color=black><br>
# 
# - What happens when I give a sentence with strange words (and stop words), and try to get its word vector in Spacy?
# - Well, at least, this is better than throwing an exception!
# 
# <br></font>
# </div>

# In[24]:


temp = nlp('practicalnlp is a newword')
temp[0].vector


# # References

# <div class="alert alert-block alert-warning">
# <font color=black><br>
# 
# - https://github.com/practical-nlp/practical-nlp/blob/master/Ch3/05_Pre_Trained_Word_Embeddings.ipynb
# 
# <br></font>
# </div>

# In[ ]:




