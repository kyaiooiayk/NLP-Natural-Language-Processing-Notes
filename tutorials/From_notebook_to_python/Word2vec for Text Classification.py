#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/word2vec_text_classification.ipynb


# # Import modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# change default style figure and font size
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['font.size'] = 12


# # Leveraging Word2vec for Text Classification

# Many machine learning algorithms requires the input features to be represented as a fixed-length feature
# vector. When it comes to texts, one of the most common fixed-length features is one hot encoding methods such as bag of words or tf-idf. The advantage of these approach is that they have fast execution time, while the main drawback is they lose the ordering & semantics of the words.
# 
# The motivation behind converting text into semantic vectors (such as the ones provided by `Word2Vec`) is that not only do these type of methods have the capabilities to extract the semantic relationships (e.g. the word powerful should be closely related to strong as oppose to another word like bank), but they should be preserve most of the relevant information about a text while having relatively low dimensionality.
# 
# In this notebook, we'll take a look at how a `Word2Vec` model can also be used as a dimensionality reduction algorithm to feed into a text classifier. A good one should be able to extract the signal from the noise efficiently, hence improving the performance of the classifier.

# ## Data Preparation

# We'll download the text classification data, read it into a `pandas` dataframe and split it into train and test set.

# In[3]:


import os
from subprocess import call


def download_data(base_dir='.'):
    """download Reuters' text categorization benchmarks from its url."""
    
    train_data = 'r8-train-no-stop.txt'
    test_data = 'r8-test-no-stop.txt'
    concat_data = 'r8-no-stop.txt'
    base_url = 'http://www.cs.umb.edu/~smimarog/textmining/datasets/'
    
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        
    dir_prefix_flag = ' --directory-prefix ' + base_dir

    # brew install wget
    # on a mac if you don't have it
    train_data_path = os.path.join(base_dir, train_data)
    if not os.path.isfile(train_data_path):
        call('wget ' + base_url + train_data + dir_prefix_flag, shell=True)
    
    test_data_path = os.path.join(base_dir, test_data)
    if not os.path.isfile(test_data_path):
        call('wget ' + base_url + test_data + dir_prefix_flag, shell=True)

    concat_data_path = os.path.join(base_dir, concat_data)
    if not os.path.isfile(concat_data_path):
        # concatenate train and test files, we'll make our own train-test splits
        # the > piping symbol directs the concatenated file to a new file, it
        # will replace the file if it already exists; on the other hand, the >> symbol
        # will append if it already exists
        train_test_path = os.path.join(base_dir, 'r8-*-no-stop.txt')
        call('cat {} > {}'.format(train_test_path, concat_data_path), shell=True)

    return concat_data_path


# In[4]:


base_dir = 'data'
data_path = download_data(base_dir)
data_path


# In[5]:


def load_data(data_path):
    texts, labels = [], []
    with open(data_path) as f:
        for line in f:
            label, text = line.split('\t')
            # texts are already tokenized, just split on space
            # in a real use-case we would put more effort in preprocessing
            texts.append(text.split())
            labels.append(label)
            
    return pd.DataFrame({'texts': texts, 'labels': labels})


# In[6]:


data = load_data(data_path)
data['labels'] = data['labels'].astype('category')
print('dimension: ', data.shape)
data.head()


# In[7]:


label_mapping = data['labels'].cat.categories
data['labels'] = data['labels'].cat.codes
X = data['texts']
y = data['labels']


# In[8]:


test_size = 0.1
random_state = 1234

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y)

# val_size = 0.1
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)


# ## Gensim Implementation

# After feeding the `Word2Vec` algorithm with our corpus, it will learn a vector representation for each word. This by itself, however, is still not enough to be used as features for text classification as each record in our data is a document not a word.
# 
# To extend these word vectors and generate document level vectors, we'll take the naive approach and use an average of all the words in the document (We could also leverage tf-idf to generate a weighted-average version, but that is not done here). The `Word2Vec` algorithm is wrapped inside a sklearn-compatible transformer which can be used almost the same way as `CountVectorizer` or `TfidfVectorizer` from `sklearn.feature_extraction.text`. Almost - because sklearn vectorizers can also do their own tokenization - a feature which we won't be using anyway because the corpus we will be using is already tokenized.
# 
# In the next few code chunks, we will build a pipeline that transforms the text into low dimensional vectors via average word vectors as use it to fit a boosted tree model, we then report the performance of the training/test set.
# 
# The `transformers` folder that contains the implementation is at the following [link](https://github.com/ethen8181/machine-learning/tree/master/keras/text_classification/transformers).

# In[9]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from transformers import GensimWord2VecVectorizer

gensim_word2vec_tr = GensimWord2VecVectorizer(size=50, min_count=3, sg=1, alpha=0.025, iter=10)
xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
w2v_xgb = Pipeline([
    ('w2v', gensim_word2vec_tr), 
    ('xgb', xgb)
])
w2v_xgb


# In[10]:


import time

start = time.time()
w2v_xgb.fit(X_train, y_train)
elapse = time.time() - start
print('elapsed: ', elapse)
w2v_xgb


# In[11]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_train_pred = w2v_xgb.predict(X_train)
print('Training set accuracy %s' % accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


# In[12]:


y_test_pred = w2v_xgb.predict(X_test)
print('Test set accuracy %s' % accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)


# We can extract the Word2vec part of the pipeline and do some sanity check of whether the word vectors that were learned made any sense.

# In[13]:


vocab_size = len(w2v_xgb.named_steps['w2v'].model_.wv.index2word)
print('vocabulary size:', vocab_size)
w2v_xgb.named_steps['w2v'].model_.wv.most_similar(positive=['stock'])


# ## Keras Implementation

# We'll also show how we can use a generic deep learning framework to implement the `Wor2Vec` part of the pipeline. There are many variants of `Wor2Vec`, here, we'll only be implementing skip-gram and negative sampling.
# 
# The flow would look like the following:
# 
# An (integer) input of a target word and a real or negative context word. This is essentially the skipgram part where any word within the context of the target word is a real context word and we randomly draw from the rest of the vocabulary to serve as the negative context words.
# 
# An embedding layer lookup (i.e. looking up the integer index of the word in the embedding matrix to get the word vector).
# 
# A dot product operation. As the network trains, words which are similar should end up having similar embedding vectors. The most popular way of measuring similarity between two vectors $A$ and $B$ is the cosine similarity.
# 
# \begin{align}
# similarity = cos(\theta) = \frac{\textbf{A}\cdot\textbf{B}}{\parallel\textbf{A}\parallel_2 \parallel \textbf{B} \parallel_2}
# \end{align}
# 
# The denominator of this measure acts to normalize the result – the real similarity operation is on the numerator: the dot product between vectors $A$ and $B$.
# 
# Followed by a sigmoid output layer. Our network is a binary classifier since it's distinguishing words from the same context versus those that aren't.

# In[14]:


# the keras model/graph would look something like this:
from keras import layers, optimizers, Model

# adjustable parameter that control the dimension of the word vectors
embed_size = 100

input_center = layers.Input((1,))
input_context = layers.Input((1,))

embedding = layers.Embedding(vocab_size, embed_size, input_length=1, name='embed_in')
center = embedding(input_center)  # shape [seq_len, # features (1), embed_size]
context = embedding(input_context)

center = layers.Reshape((embed_size,))(center)
context = layers.Reshape((embed_size,))(context)

dot_product = layers.dot([center, context], axes=1)
output = layers.Dense(1, activation='sigmoid')(dot_product)
model = Model(inputs=[input_center, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.01))
model.summary()


# In[15]:


# then we can feed in the skipgram and its label (whether the word pair is in or outside
# the context)
batch_center = [2354, 2354, 2354, 69, 69]
batch_context = [4288, 203, 69, 2535, 815]
batch_label = [0, 1, 1, 0, 1]
model.train_on_batch([batch_center, batch_context], batch_label)


# The `transformers` folder that contains the implementation is at the following [link](https://github.com/ethen8181/machine-learning/tree/master/keras/text_classification/transformers).

# In[16]:


from transformers import KerasWord2VecVectorizer

keras_word2vec_tr = KerasWord2VecVectorizer(embed_size=50, min_count=3, epochs=5000,
                                            negative_samples=2)
keras_word2vec_tr


# In[17]:


keras_w2v_xgb = Pipeline([
    ('w2v', keras_word2vec_tr), 
    ('xgb', xgb)
])

keras_w2v_xgb.fit(X_train, y_train)


# In[18]:


y_train_pred = keras_w2v_xgb.predict(X_train)
print('Training set accuracy %s' % accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


# In[19]:


y_test_pred = keras_w2v_xgb.predict(X_test)
print('Test set accuracy %s' % accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)


# In[20]:


print('vocabulary size:', keras_w2v_xgb.named_steps['w2v'].vocab_size_)
keras_w2v_xgb.named_steps['w2v'].most_similar(positive=['stock'])


# ## Benchmarks

# We'll compare the word2vec + xgboost approach with tfidf + logistic regression. The latter approach is known for its interpretability and fast training time, hence serves as a strong baseline.
# 
# Note that for sklearn's tfidf, we didn't use the default analyzer 'words', as this means it expects that input is a single string which it will try to split into individual words, but our texts are already tokenized, i.e. already lists of words.

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
 
tfidf = TfidfVectorizer(stop_words='english', analyzer=lambda x: x)
logistic = LogisticRegression(solver='liblinear', multi_class='auto')

tfidf_logistic = Pipeline([
    ('tfidf', tfidf), 
    ('logistic', logistic)
])


# In[22]:


from scipy.stats import randint, uniform

w2v_params = {'w2v__size': [100, 150, 200]}
tfidf_params = {'tfidf__ngram_range': [(1, 1), (1, 2)]}
logistic_params = {'logistic__C': [0.5, 1.0, 1.5]}
xgb_params = {'xgb__max_depth': randint(low=3, high=12),
              'xgb__colsample_bytree': uniform(loc=0.8, scale=0.2),
              'xgb__subsample': uniform(loc=0.8, scale=0.2)}

tfidf_logistic_params = {**tfidf_params, **logistic_params}
w2v_xgb_params = {**w2v_params, **xgb_params}


# In[23]:


from sklearn.model_selection import RandomizedSearchCV

cv = 3
n_iter = 3
random_state = 1234
scoring = 'accuracy'

all_models = [
    ('w2v_xgb', w2v_xgb, w2v_xgb_params),
    ('tfidf_logistic', tfidf_logistic, tfidf_logistic_params)
]

all_models_info = []
for name, model, params in all_models:
    print('training:', name)
    model_tuned = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=cv,
        n_iter=n_iter,
        n_jobs=-1,
        verbose=1,
        scoring=scoring,
        random_state=random_state,
        return_train_score=False
    ).fit(X_train, y_train)
    
    y_test_pred = model_tuned.predict(X_test)
    test_score = accuracy_score(y_test, y_test_pred)
    info = name, model_tuned.best_score_, test_score, model_tuned
    all_models_info.append(info)

columns = ['model_name', 'train_score', 'test_score', 'estimator']
results = pd.DataFrame(all_models_info, columns=columns)
results = (results
           .sort_values('test_score', ascending=False)
           .reset_index(drop=True))
results


# Note that different run may result in different performance being reported. And as our dataset changes, different approaches might that worked the best on one dataset might no longer be the best. Especially since the dataset we're working with here isn't very big, training an embedding from scratch will most likely not reach its full potential.
# 
# There are many other text classification techniques in the deep learning realm that we haven't yet explored, we'll leave that for another day.

# # Reference

# - [Blog: A Word2Vec Keras tutorial](https://adventuresinmachinelearning.com/word2vec-keras-tutorial/)
# - [Blog: Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
