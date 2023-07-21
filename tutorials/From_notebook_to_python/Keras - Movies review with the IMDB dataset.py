#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Movie review with the IMDB dataset
# 
# </font>
# </div>

# # Theoretical recall: binary classification
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Binary classification
# - How to classify movie reviews as positive or negative, based on the text content of the reviews.-
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[41]:


import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 14, 5
rcParams['font.size'] = 20


# # Load the datatest
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - 50,000 highly polarized reviews
# - They’re split into 25,000 reviews for training and 25,000 reviews for testing, each set consisting of 50% negative and 50% positive reviews.
# - 80 MB of data will be downloaded to your machine
# - num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data. Rare words will be discarded. This allows you to work with vector data of manageable size.
# - The variables train_data and test_data are lists of reviews; each review is a list of word indices (encoding a sequence of words).
# 
# </font>
# </div>

# In[ ]:


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)


# In[5]:


train_data[0]


# In[17]:


print(type(train_data))
print(train_data.ndim)
print(train_data.shape)


# In[7]:


train_labels


# In[8]:


# We are using only 10k most frequent words
max([max(sequence) for sequence in train_data])


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Here’s how you can quickly decode one of these reviews back to English words. Did not get it!
# 
# </font>
# </div>

# In[9]:


word_index = imdb.get_word_index()
# Reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Decodes the reviews. Note that the indices are offset by 3 because 0,1 and 2 
# are reserved indices for padding
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[13]:


decoded_review


# # Pre-processing
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - DDNs accept tensor as an input, when this is not the case you have the create one
# - Encoding the integer sequences into a binary matrix
# - One-hot encode your lists to turn them into vectors of 0s and 1s. 
# - This would mean, for instance, turning the sequence [3, 5] into a 10,000-dimensional vector that would be all 0s except for indices 3 and 5, which would be 1s.
# 
# </font>
# </div>

# In[18]:


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
          results[i, sequence] = 1.
    return results


# In[19]:


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[24]:


x_train


# In[21]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# In[23]:


y_train


# # Building the NN
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The NN consists of 16 hidden simple stack of fully connected (dense) layers with RELUs
# - 16 hidden units means the weight matrix W will have shape (input_dimension, 16)
# - The dot product with W will project the input data onto a 16-dimensional representation space 
# - Then you’ll add the bias vector b and apply the relu operation).
# - The intermediate layers will use relu as their activation function
# - The final layer will use a sigmoid activation so as to output a probability (a score between 0 and 1)
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# In[26]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# # Compiling the model
# <hr style="border:2px solid black"> </hr>

# In[30]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[31]:


# Alternatively you can also configure the optimiser directly
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])


# In[32]:


# another alternative is
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                      loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - In order to **monitor** during training the accuracy of the model on data it has never seen before, you’ll create a validation set by setting apart 10,000 samples **from** the original training data.
# 
# </font>
# </div>

# In[35]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# # Training the model
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Train the model for 20 epochs (20 iterations over all samples in the x_train and y_train tensors)
# - Use mini-batches of 512 samples
# - Monitor loss and accuracy on the 10,000 samples that you set apart. 
# 
# </font>
# </div>

# In[36]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# In[37]:


history_dict = history.history
history_dict.keys()


# # Post-processing
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The **training** loss decreases with every epoch, and the training accuracy increases with every epoch.
# - The **validation** loss and accurcy follow a different trend: they peak at the 4th epoch
# - This is classical case of **overfitting**
# - To avoid this you can (other options are available) stop the training at the 4th epoch
# 
# </font>
# </div>

# In[47]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, history_dict['acc'], 'bo', label='Training acc')
plt.plot(epochs, history_dict['val_acc'], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # Avoiding overfitting
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - This fairly naive approach achieves an accuracy of 88%. 
# - With state-of-the-art approaches, you should be able to get close to 95%.
# 
# </font>
# </div>

# In[48]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[49]:


results


# # Predicting on new data
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - As you can see, the network is confident for some samples (0.99 or more, or 0.01 or
# less) but less confident for others (0.6, 0.4).
# 
# </font>
# </div>

# In[51]:


model.predict(x_test)


# # Conclusions
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-danger">
# <font color=black>
# 
# - There is quite a lot of pre-processng to do
# - Dense + relu NN are pretty flexible
# - In binary classification, the network should end with a sigmoid activation function
# - WIn binary classification, the loss function function soulf be: binayr_crossentropy
# - RMSprop is pretty good whatever your problem
# - Look out for overfitting
# 
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/fchollet/deep-learning-with-python-notebooks
# - Chollet, Francois. Deep learning with Python. Vol. 361. New York: Manning, 2018
# 
# </font>
# </div>

# In[ ]:




