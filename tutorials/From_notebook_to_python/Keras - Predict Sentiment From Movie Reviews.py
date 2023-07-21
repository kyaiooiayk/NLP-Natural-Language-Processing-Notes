#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-the-data-set-&amp;-analysis" data-toc-modified-id="Load-the-data-set-&amp;-analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Load the data set &amp; analysis</a></span></li><li><span><a href="#Simple-Multilayer-Perceptron-Model" data-toc-modified-id="Simple-Multilayer-Perceptron-Model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Simple Multilayer Perceptron Model</a></span></li><li><span><a href="#One-Dimensional-Convolutional-NN" data-toc-modified-id="One-Dimensional-Convolutional-NN-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>One-Dimensional Convolutional NN</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Predict Sentiment From Movie Reviews
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[19]:


import numpy
from keras.datasets import imdb
from matplotlib import pyplot
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D


# # Load the data set & analysis
# <hr style="border:2px solid black"> </hr>

# In[4]:


# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)


# In[10]:


# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)

# Summarize number of classes
print("Classes: ")
print(numpy.unique(y))

# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))

# Summarize review length
print("Review length: ")
result = list(map(len, X))
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))



# In[11]:


# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()


# In[ ]:


"""
Looking at the box and whisker plot and the histogram for the review lengths in words, we can probably see 
an exponential distribution that we can probably cover the mass of the distribution with a clipped length 
of 400 to 500 words
"""


# # Simple Multilayer Perceptron Model
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
A recent breakthrough in the field of natural language processing is called word embedding. This is a technique 
where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words 
in terms of meaning translates to closeness in the vector space.
"""


# In[17]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation= "relu" ))
model.add(Dense(1, activation= "sigmoid" ))
model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=["accuracy"])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
I’m sure we can do better if we trained this network, perhaps using a larger embedding and adding more hidden 
layers. Let’s try a di↵erent network type.
"""


# # One-Dimensional Convolutional NN
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
Convolutional neural networks were designed to honor the spatial structure in image data whilst being robust to
the position and orientation of learned objects in the scene. This same princip can be used on sequences, such 
as the one-dimensional sequence of words in a movie review. The same properties that make the CNN model 
attractive for learning to recognize objects in images can help to learn structure in paragraphs of words,
namely the techniques invariance to the specific position of features
"""


# In[25]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))

#model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode= "same" ,activation= "relu" ))
model.add(Convolution1D(filters =32, kernel_size=3, padding = "same", activation= "relu" ))

#model.add(MaxPooling1D(pool_length=2))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(250, activation= "relu" ))
model.add(Dense(1, activation= "sigmoid" ))
model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=["accuracy"])
print(model.summary())
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128,verbose=1)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
Again, there is a lot of opportunity for further optimization, such as the use of deeper and/or larger 
convolutional layers. One interesting idea is to set the max pooling layer to use an input length of 500. 
This would compress each feature map to a single 32 length vector and may boost performance.
"""


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
# - https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# 
# </font>
# </div>

# In[ ]:




