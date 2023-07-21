#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Movie genres classification with KERAS

NLTK is a standard python library for natural language processing and computational linguistics.

Reference: https://www.mygreatlearning.com/blog/nltk-tutorial-with-python/?highlight=nlp
"""


# # Import modules

# In[1]:


import numpy as np
from keras.utils import to_categorical 
from keras import models 
from keras import layers 
from keras.datasets import imdb


# # Import dataset and split the data

# In[4]:


(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=10000) 
dt = np.concatenate((train_data, test_data), axis=0) 
tar = np.concatenate((train_target, test_target), axis=0) 

# Train and test dataset
dt = convert(dt) 
tar = np.array(tar).astype("float32")
test_x = dt[:9000] 
test_y = tar[:9000] 
train_x = dt[9000:] 
train_y = tar[9000:]
model = models.Sequential() 


# In[10]:


test_x


# In[3]:


def convert(sequences, dimension = 10000):
    """
    Convert the words into vectors for processing.
    For the sake of simplicity, we use the first 10,000 records. 
    You are free to explore with more data. The execution time 
    increases with more data.
    """
    results = np.zeros((len(sequences), dimension))  
    for i, sequence in enumerate(sequences):   
        results[i, sequence] = 1  
    return results  


# In[5]:


# Input - Layer 
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
# Hidden - Layers 
model.add(layers.Dropout(0.4, noise_shape=None, seed=None)) 
model.add(layers.Dense(50, activation = "relu")) 
model.add(layers.Dropout(0.3, noise_shape=None, seed=None)) 
model.add(layers.Dense(50, activation = "relu")) 
# Output- Layer 
model.add(layers.Dense(1, activation = "sigmoid")) 
model.summary() 


# In[6]:


# compiling the model   
model.compile(  optimizer = "adam",  loss = "binary_crossentropy",  metrics = ["accuracy"] ) 
#The output we are getting is a sparse matrix with the probability of genres most suited are returned as 1.
results = model.fit(  train_x, train_y,  epochs= 2,  batch_size = 500,  validation_data = (test_x, test_y) ) 

print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))


# In[8]:


train_x[0]


# In[9]:


train_y[0]


# In[ ]:




