#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Leveraging-Pre-trained-Word-Embedding-for-Text-Classification" data-toc-modified-id="Leveraging-Pre-trained-Word-Embedding-for-Text-Classification-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Leveraging Pre-trained Word Embedding for Text Classification</a></span><ul class="toc-item"><li><span><a href="#Data-Preparation" data-toc-modified-id="Data-Preparation-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Preparation</a></span></li><li><span><a href="#Glove" data-toc-modified-id="Glove-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Glove</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Model-with-Pretrained-Embedding" data-toc-modified-id="Model-with-Pretrained-Embedding-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Model with Pretrained Embedding</a></span></li><li><span><a href="#Model-without-Pretrained-Embedding" data-toc-modified-id="Model-without-Pretrained-Embedding-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Model without Pretrained Embedding</a></span></li></ul></li><li><span><a href="#Submission" data-toc-modified-id="Submission-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Submission</a></span></li></ul></li><li><span><a href="#Reference" data-toc-modified-id="Reference-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Reference</a></span></li></ul></div>

# # Imports

# In[ ]:


import os
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# prevent scientific notations
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# # Leveraging Pre-trained Word Embedding for Text Classification

# There are two main ways to obtain word embeddings:
# 
# - Learn it from scratch: We specify a neural network architecture and learn the word embeddings jointly with the main task at our hand (e.g. sentiment classification). i.e. we would start off with some random word embeddings, and it would update itself along with the word embeddings.
# - Transfer Learning: The whole idea behind transfer learning is to avoid reinventing the wheel as much as possible. It gives us the capability to transfer knowledge that was gained/learned in some other task and use it to improve the learning of another related task. In practice, one way to do this is for the embedding part of the neural network architecture, we load some other embeddings that were trained on a different machine learning task than the one we are trying to solve and use that to bootstrap the process.
# 
# One area that transfer learning shines is when we have little training data available and using our data alone might not be enough to learn an appropriate task specific embedding/features for our vocabulary. In this case, leveraging a word embedding that captures generic aspect of the language can prove to be beneficial from both a performance and time perspective (i.e. we won't have to spend hours/days training a model from scratch to achieve a similar performance). Keep in mind that, as with all machine learning application, everything is still all about trial and error. What makes a embedding good depends heavily on the task at hand: The word embedding for a movie review sentiment classification model may look very different from a legal document classification model as the semantic of the corpus varies between these two tasks.

# ## Data Preparation

# We'll use the movie review sentiment analysis dataset from [Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview) for this example. It's a binary classification problem with AUC as the ultimate evaluation metric. The next few code chunk performs the usual text preprocessing, build up the word vocabulary and performing a train/test split.

# In[3]:


data_dir = 'data'
submission_dir = 'submission'


# In[4]:


input_path = os.path.join(data_dir, 'word2vec-nlp-tutorial', 'labeledTrainData.tsv')
df = pd.read_csv(input_path, delimiter='\t')
print(df.shape)
df.head()


# In[5]:


raw_text = df['review'].iloc[0]
raw_text


# In[6]:


import re

def clean_str(string: str) -> str:
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


# In[7]:


from bs4 import BeautifulSoup

def clean_text(df: pd.DataFrame,
               text_col: str,
               label_col: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    for raw_text, label in zip(df[text_col], df[label_col]):  
        text = BeautifulSoup(raw_text).get_text()
        cleaned_text = clean_str(text)
        texts.append(cleaned_text)
        labels.append(label)

    return texts, labels


# In[8]:


text_col = 'review'
label_col = 'sentiment'
texts, labels = clean_text(df, text_col, label_col)
print('sample text: ', texts[0])
print('corresponding label:', labels[0])


# In[9]:


random_state = 1234
val_split = 0.2

labels = to_categorical(labels)
texts_train, texts_val, y_train, y_val = train_test_split(
    texts, labels,
    test_size=val_split,
    random_state=random_state)

print('labels shape:', labels.shape)
print('train size: ', len(texts_train))
print('validation size: ', len(texts_val))


# In[10]:


max_num_words = 20000

tokenizer = Tokenizer(num_words=max_num_words, oov_token='<unk>')
tokenizer.fit_on_texts(texts_train)
print('Found %s unique tokens.' % len(tokenizer.word_index))


# In[11]:


max_sequence_len = 1000

sequences_train = tokenizer.texts_to_sequences(texts_train)
x_train = pad_sequences(sequences_train, maxlen=max_sequence_len)

sequences_val = tokenizer.texts_to_sequences(texts_val)
x_val = pad_sequences(sequences_val, maxlen=max_sequence_len)

sequences_train[0][:5]


# ## Glove

# There are many different pretrained word embeddings online. The one we'll be using is from [Glove](https://nlp.stanford.edu/projects/glove/). Others include but not limited to [FastText](https://fasttext.cc/docs/en/crawl-vectors.html), [bpemb](https://github.com/bheinzerling/bpemb).
# 
# If we look at the project's wiki page, we can find any different pretrained embeddings available for us to experiment.
# 
# <img src="img/pretrained_weights.png" width="100%" height="100%">

# In[12]:


import requests
from tqdm import tqdm

def download_glove(embedding_type: str='glove.6B.zip'):
    """
    download GloVe word vector representations, this step may take a while
    
    Parameters
    ----------
    embedding_type : str, default 'glove.6B.zip'
        Specifying different glove embeddings to download if not already there.
        {'glove.6B.zip', 'glove.42B.300d.zip', 'glove.840B.300d.zip', 'glove.twitter.27B.zip'}
        Be wary of the size. e.g. 'glove.6B.zip' is a 822 MB zipped, 2GB unzipped
    """

    base_url = 'http://nlp.stanford.edu/data/'
    if not os.path.isfile(embedding_type):
        url = base_url + embedding_type

        # the following section is a pretty generic http get request for
        # saving large files, provides progress bars for checking progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        content_len = response.headers.get('Content-Length')
        total = int(content_len) if content_len is not None else 0

        with tqdm(unit='B', total=total) as pbar, open(embedding_type, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)

        if response.headers.get('Content-Type') == 'application/zip':
            from zipfile import ZipFile
            with ZipFile(embedding_type, 'r') as f:
                f.extractall(embedding_type.strip('.zip'))


download_glove()


# The way we'll leverage the pretrained embedding is to first read it in as a dictionary lookup, where the key is the word and the value is the corresponding word embedding. Then for each token in our vocabulary, we'll lookup this dictionary to see if there's a pretrained embedding available, if there is, we'll use the pretrained embedding, if there isn't, we'll leave the embedding for this word in its original randomly initialized form.
# 
# The format for this particular pretrained embedding is for every line, we have a space delimited values, where the first token is the word, and the rest are its corresponding embedding values. e.g. the first line from the line looks like:
# 
# ```
# the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459 0.28787 -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336 0.1464 -0.37304 0.34577 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107 -0.039722 0.28277 0.14393 0.23464 -0.31021 0.086173 0.20397 0.52624 0.17164 -0.082378 -0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 -0.33477 -0.36559 -0.54857 -0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 0.72201 -0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044 0.057162 -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624 0.70565 0.49744 -0.42063 0.26148 -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217 0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 -0.1459 0.8278 0.27062
# ```

# In[13]:


def get_embedding_lookup(embedding_path) -> Dict[str, np.ndarray]:
    embedding_lookup = {}
    with open(embedding_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coef = np.array(values[1:], dtype=np.float32)
            embedding_lookup[word] = coef

    return embedding_lookup


def get_pretrained_embedding(embedding_path: str,
                             index2word: Dict[int, str],
                             max_features: int) -> np.ndarray:
    embedding_lookup = get_embedding_lookup(embedding_path)

    pretrained_embedding = np.stack(list(embedding_lookup.values()))
    embedding_dim = pretrained_embedding.shape[1]
    embeddings = np.random.normal(pretrained_embedding.mean(),
                                  pretrained_embedding.std(),
                                  (max_features, embedding_dim)).astype(np.float32)
    # we track how many tokens in our vocabulary exists in the pre-trained embedding,
    # i.e. how many tokens has a pre-trained embedding from this particular file
    n_found = 0
    
    # the loop starts from 1 due to keras' Tokenizer reserves 0 for padding index
    for i in range(1, max_features):
        word = index2word[i]
        embedding_vector = embedding_lookup.get(word)
        if embedding_vector is not None:
            embeddings[i] = embedding_vector
            n_found += 1

    print('number of words found:', n_found)
    return embeddings


# In[14]:


glove_path = os.path.join('glove.6B', 'glove.6B.100d.txt')
max_features = max_num_words + 1

pretrained_embedding = get_pretrained_embedding(glove_path, tokenizer.index_word, max_features)
pretrained_embedding.shape


# ## Model

# To train our text classifier, we specify a 1D convolutional network. Our embedding layer can either be initialized randomly or loaded from a pre-trained embedding. Note that for the pre-trained embedding case, apart from loading the weights, we also "freeze" the embedding layer, i.e. we set its trainable attribute to False. This idea is often times used in transfer learning, where when parts of a model are pre-trained (in our case, only our Embedding layer), and parts of it are randomly initialized, the pre-trained part should ideally not be trained together with the randomly initialized part. The rationale behind it is that a large gradient update triggered by the randomly initialized layer would become very disruptive to those pre-trained weights.
# 
# Once we train the randomly initialized weights for a few iterations, we can then go about un-freezing the layers that were loaded with pre-trained weights, and do an update on the weight for the entire thing. The [keras documentation](https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes) also provides an example of how to do this, although the example is for image models, the same idea can also be applied here, and can be something that's worth experimenting.

# In[ ]:


def simple_text_cnn(max_sequence_len: int,
                    max_features: int,
                    num_classes: int,
                    optimizer: str='adam',
                    metrics: List[str]=['acc'],
                    pretrained_embedding: np.ndarray=None) -> Model:

    sequence_input = layers.Input(shape=(max_sequence_len,), dtype='int32')
    if pretrained_embedding is None:
        embedded_sequences = layers.Embedding(max_features, 100,
                                              name='embedding')(sequence_input)
    else:
        embedded_sequences = layers.Embedding(max_features, pretrained_embedding.shape[1],
                                              weights=[pretrained_embedding],
                                              name='embedding',
                                              trainable=False)(sequence_input)

    conv1 = layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
    pool1 = layers.MaxPooling1D(5)(conv1)
    conv2 = layers.Conv1D(128, 5, activation='relu')(pool1)
    pool2 = layers.MaxPooling1D(5)(conv2)
    conv3 = layers.Conv1D(128, 5, activation='relu')(pool2)
    pool3 = layers.MaxPooling1D(35)(conv3)
    flatten = layers.Flatten()(pool3)
    dense = layers.Dense(128, activation='relu')(flatten)
    preds = layers.Dense(num_classes, activation='softmax')(dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)
    return model


# ### Model with Pretrained Embedding

# In[17]:


num_classes = 2
model1 = simple_text_cnn(max_sequence_len, max_features, num_classes,
                         pretrained_embedding=pretrained_embedding)
model1.summary()


# We can confirm whether our embedding layer is trainable by looping through each layer and checking the trainable attribute.

# In[18]:


df_model_layers = pd.DataFrame(
    [(layer.name, layer.trainable, layer.count_params()) for layer in model1.layers],
    columns=['layer', 'trainable', 'n_params']
)
df_model_layers


# In[19]:


# time : 70
# test performance : auc 0.93212
start = time.time()
history1 = model1.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=128,
                      epochs=8)
end = time.time()
elapse1 = end - start
elapse1


# ### Model without Pretrained Embedding

# In[20]:


num_classes = 2
model2 = simple_text_cnn(max_sequence_len, max_features, num_classes)
model2.summary()


# In[21]:


# time : 86 secs
# test performance : auc 0.92310
start = time.time()
history1 = model2.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=128,
                      epochs=8)
end = time.time()
elapse1 = end - start
elapse1


# ## Submission

# For the submission section, we read in and preprocess the test data provided by the competition, then generate the predicted probability column for both the model that uses pretrained embedding and one that doesn't to compare their performance.

# In[22]:


input_path = os.path.join(data_dir, 'word2vec-nlp-tutorial', 'testData.tsv')
df_test = pd.read_csv(input_path, delimiter='\t')
print(df_test.shape)
df_test.head()


# In[ ]:


def clean_text_without_label(df: pd.DataFrame, text_col: str) -> List[str]:
    texts = []
    for raw_text in df[text_col]:
        text = BeautifulSoup(raw_text).get_text()
        cleaned_text = clean_str(text)
        texts.append(cleaned_text)

    return texts


# In[25]:


texts_test = clean_text_without_label(df_test, text_col)
sequences_test = tokenizer.texts_to_sequences(texts_test)
x_test = pad_sequences(sequences_test, maxlen=max_sequence_len)
len(x_test)


# In[ ]:


def create_submission(ids, predictions, ids_col, prediction_col, submission_path) -> pd.DataFrame:
    df_submission = pd.DataFrame({
        ids_col: ids,
        prediction_col: predictions
    }, columns=[ids_col, prediction_col])

    if submission_path is not None:
        # create the directory if need be, e.g. if the submission_path = submission/submission.csv
        # we'll create the submission directory first if it doesn't exist
        directory = os.path.split(submission_path)[0]
        if (directory != '' or directory != '.') and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        df_submission.to_csv(submission_path, index=False, header=True)

    return df_submission


# In[27]:


ids_col = 'id'
prediction_col = 'sentiment'
ids = df_test[ids_col]

models = {
    'pretrained_embedding': model1,
    'without_pretrained_embedding': model2
}

for model_name, model in models.items():
    print('generating submission for: ', model_name)
    submission_path = os.path.join(submission_dir, '{}_submission.csv'.format(model_name))
    predictions = model.predict(x_test, verbose=1)[:, 1]
    df_submission = create_submission(ids, predictions, ids_col, prediction_col, submission_path)

# sanity check to make sure the size and the output of the submission makes sense
print(df_submission.shape)
df_submission.head()


# # Reference

# - http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_pretrained_embedding.ipynb
# - [Blog: Text Classification, Part I - Convolutional Networks](https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/)
# - [Blog: Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
# - [Jupyter Notebook - Deep Learning with Python - Using Word Embeddings](https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb)

# In[ ]:




