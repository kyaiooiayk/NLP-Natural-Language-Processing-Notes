#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Simple-LSTM-for-Sequence-Classification" data-toc-modified-id="Simple-LSTM-for-Sequence-Classification-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Simple LSTM for Sequence Classification</a></span></li><li><span><a href="#LSTM-For-Sequence-Classification-With-Dropout" data-toc-modified-id="LSTM-For-Sequence-Classification-With-Dropout-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>LSTM For Sequence Classification With Dropout</a></span></li><li><span><a href="#LSTM-and-CNN-For-Sequence-Classification" data-toc-modified-id="LSTM-and-CNN-For-Sequence-Classification-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>LSTM and CNN For Sequence Classification</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Movie reviews via deep learning
# 
# </font>
# </div>

# In[ ]:


"""
Sequence classification is a predictive modeling problem where you have some sequence of inputs over space or 
time and the task is to predict a category for the sequence. What makes this problem di cult is that the
sequences can vary in length, be comprised of a very large vocabulary of input symbols and may require the
model to learn the long term context or dependencies between symbols in the input sequence
"""


# # Imports
# <hr style="border:2px solid black"> </hr>

# In[19]:


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers import MaxPooling1D


# # Simple LSTM for Sequence Classification
# <hr style="border:2px solid black"> </hr>

# In[4]:


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation= "sigmoid"))
model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs = 3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
You can see that this simple LSTM with little tuning achieves near state-of-the-art results on the IMDB problem.
"""


# # LSTM For Sequence Classification With Dropout
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
Recurrent Neural networks like LSTM generally have the problem of overfitting. Dropout can be applied between 
layers using
"""


# In[11]:


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs = 3, batch_size = 64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
We can see dropout having the DESIDER IMPACT on training with a slightly slower trend in convergence and in 
this case a lower final accuracy. The model could probably use a few more epochs of training and may achieve 
a higher skill (try it an see). Alternately, dropout can be applied to the input and recurrent connections of
the memory units with the LSTM precisely and separately.


Keras provides this capability with parameters on the LSTM layer, the dropout W for configuring the input 
dropout and dropout U for configuring the recurrent dropout. For example, we can modify the first example 
to add dropout to the input and recurrent connections as follows:
"""


# In[13]:


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# old API --->>> model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation = "sigmoid"))
model.compile(loss= "binary_crossentropy" , optimizer= "adam" , metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs = 3, batch_size = 64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
ATTENTION: please check with the new API that everything is OK!

We can see that the LSTM specific dropout has a more pronounced e↵ect on the convergence of the network than 
the layer-wise dropout. As above, the number of epochs was kept constant and could be increased to see if the
skill of the model can be further lifted. Dropout is a powerful technique for combating overfitting in your 
LSTM models and it is a good idea to try both methods, 
"""


# # LSTM and CNN For Sequence Classification
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
Convolutional neural networks excel at learning the spatial structure in input data. The IMDB review data does 
have a one-dimensional spatial structure in the sequence of words in reviews and the CNN may be able to pick out
invariant features for good and bad sentiment. This learned spatial features may then be learned as sequences 
by an LSTM layer. We can easily add a one-dimensional CNN and max pooling layers after the Embedding layer which
then feed the consolidated features to the LSTM
"""


# In[22]:


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length,
          input_length=max_review_length))
#model.add(Convolution1D(nb_filter=32, filter_length=3, padding = "same", activation= "relu"))
model.add(Convolution1D(filters=32, kernel_size=3,
          padding="same", activation="relu"))

# model.add(MaxPooling1D(pool_length=2))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


"""
We can see that we achieve similar results to the first example although with less weights and faster training 
time. 
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




