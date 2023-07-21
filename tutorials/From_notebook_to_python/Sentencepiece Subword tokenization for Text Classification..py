#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_subword_tokenization.ipynb


# # Import modules

# In[1]:


import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from keras import layers
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


# # Subword Tokenization for Text Classification

# In this notebook, we will be experimenting with subword tokenization. Tokenization is often times one of the first mandatory task that's performed in NLP task, where we break down a piece of text into meaningful individual units/tokens.
# 
# There're three major ways of performing tokenization.
# 
# **Character Level**
# 
# Treats each character (or unicode) as one individual token.
# 
# - Pros: This one requires the least amount of preprocessing techniques.
# - Cons: The downstream task needs to be able to learn relative positions of the characters, dependencies, spellings, making it harder to achieve good performance.
# 
# **Word Level**
# 
# Performs word segmentation on top of our text data.
# 
# - Pros: Words are how we as human process text information.
# - Cons: The correctness of the segmentation is highly dependent on the software we're using. e.g. [Spacy's Tokenization](https://spacy.io/usage/spacy-101#annotations-token) performs language specific rules to segment the original text into words. Also word level can't handle unseen words (a.k.a. out of vocabulary words) and performs poorly on rare words.
# 
# [Blog: Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html) also shared some thoughts comparing character based tokenization v.s. word based tokenization. Taken directly from the post.
# 
# > Word-level models have an important advantage over char-level models. Take the following sequence as an example (a quote from Robert A. Heinlein):
# >
# > Progress isn't made by early risers. It's made by lazy men trying to find easier ways to do something.
# >
# > After tokenization, the word-level model might view this sequence as containing 22 tokens. On the other hand, the char-level will view this sequence as containing 102 tokens. This longer sequence makes the task of the character model harder than the word model, as it must take into account dependencies between more tokens over more time-steps. Another issue with character language models is that they need to learn spelling in addition to syntax, semantics, etc. In any case, word language models will typically have lower error than character models.
# >
# > The main advantage of character over word language models is that they have a really small vocabulary. For example, the GBW dataset will contain approximately 800 characters compared to 800,000 words (after pruning low-frequency tokens). In practice this means that character models will require less memory and have faster inference than their word counterparts. Another advantage is that they do not require tokenization as a preprocessing step.
# 
# **Subword Level**
# 
# As we can probably imagine, subword level is somewhere between character level and word level, hence tries to bring in the the pros (being able to handle out of vocabulary or rare words better) and mitigate the drawback (too fine-grained for downstream tasks) from both approaches. With subword level, what we are aiming for is to represent open vocabulary through a fixed-sized vocabulary of variable length character sequences. e.g. the word highest might be segmented into subwords high and est.
# 
# There're many different methods for generating these subwords. e.g.
# 
# - A naive way way is to brute force generate the subwords by sliding through a fix sized window. e.g. highest -> hig, igh, ghe, etc.
# - More clever approaches such as Byte Pair Encoding, Unigram models. We won't be covering the internals of these approaches here. There's another [document](https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/subword/bpe.ipynb) that goes more in-depth into Byte Pair Encoding and sentencepiece, the open-sourced package that we'll be using here to experiment with subword tokenization.

# ## Data Preprocessing

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


# ## Model

# To train our text classifier, we specify a 1D convolutional network. The comparison we'll be experimenting is whether subword-level model gives a better performance than word-level model.

# In[10]:


def simple_text_cnn(max_sequence_len: int, max_features: int, num_classes: int,
                    optimizer: str='adam', metrics: List[str]=['acc']) -> Model:

    sequence_input = layers.Input(shape=(max_sequence_len,), dtype='int32')
    embedded_sequences = layers.Embedding(max_features, 100,
                                          trainable=True)(sequence_input)
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


# ### Subword-Level Tokenizer

# The next couple of code chunks trains the subword vocabulary, encode our original text into these subwords and pads the sequences into a fixed length.
# 
# Note the the `pad_sequences` function from keras assumes that index 0 is reserved for padding, hence when learning the subword vocabulary using `sentencepiece`, we make sure to keep the index consistent.

# In[11]:


# write the raw text so that sentencepiece can consume it
temp_file = 'train.txt'
with open(temp_file, 'w') as f:
    f.write('\n'.join(texts))


# In[12]:


from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

max_num_words = 30000
model_type = 'unigram'
model_prefix = model_type
pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3

sentencepiece_params = ' '.join([
    '--input={}'.format(temp_file),
    '--model_type={}'.format(model_type),
    '--model_prefix={}'.format(model_type),
    '--vocab_size={}'.format(max_num_words),
    '--pad_id={}'.format(pad_id),
    '--unk_id={}'.format(unk_id),
    '--bos_id={}'.format(bos_id),
    '--eos_id={}'.format(eos_id)
])
print(sentencepiece_params)
SentencePieceTrainer.train(sentencepiece_params)


# In[13]:


sp = SentencePieceProcessor()
sp.load("{}.model".format(model_prefix))
print('Found %s unique tokens.' % sp.get_piece_size())


# In[14]:


max_sequence_len = 1000

sequences_train = [sp.encode_as_ids(text) for text in texts_train]
x_train = pad_sequences(sequences_train, maxlen=max_sequence_len)

sequences_val = [sp.encode_as_ids(text) for text in texts_val]
x_val = pad_sequences(sequences_val, maxlen=max_sequence_len)

sequences_train[0][:5]


# In[15]:


print('sample text: ', texts_train[0])
print('sample text: ', sp.encode_as_pieces(sp.decode_ids(x_train[0].tolist())))


# In[ ]:


num_classes = 2
model1 = simple_text_cnn(max_sequence_len, max_num_words + 1, num_classes)
model1.summary()


# In[ ]:


# time : 120
# performance : 0.92936
start = time.time()
history1 = model1.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=128,
                      epochs=8)
end = time.time()
elapse1 = end - start
elapse1


# ### Word-Level Tokenizer

# In[ ]:


tokenizer = Tokenizer(num_words=max_num_words, oov_token='<unk>')
tokenizer.fit_on_texts(texts_train)
print('Found %s unique tokens.' % len(tokenizer.word_index))


# In[ ]:


sequences_train = tokenizer.texts_to_sequences(texts_train)
x_train = pad_sequences(sequences_train, maxlen=max_sequence_len)

sequences_val = tokenizer.texts_to_sequences(texts_val)
x_val = pad_sequences(sequences_val, maxlen=max_sequence_len)


# In[ ]:


num_classes = 2
model2 = simple_text_cnn(max_sequence_len, max_num_words + 1, num_classes)
model2.summary()


# In[ ]:


# time : 120
# performance : 0.92520
start = time.time()
history2 = model2.fit(x_train, y_train,
                      validation_data=(x_val, y_val),
                      batch_size=128,
                      epochs=8)
end = time.time()
elapse2 = end - start
elapse2


# ## Submission

# For the submission section, we read in and preprocess the test data provided by the competition, then generate the predicted probability column for both the model that uses word-level tokenization and one that uses subword tokenization to compare their performance.

# In[ ]:


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


# In[ ]:


texts_test = clean_text_without_label(df_test, text_col)

# word-level
word_sequences_test = tokenizer.texts_to_sequences(texts_test)
word_x_test = pad_sequences(word_sequences_test, maxlen=max_sequence_len)
len(word_x_test)


# In[ ]:


# subword-level
sentencepiece_sequences_test = [sp.encode_as_ids(text) for text in texts_test]
sentencepiece_x_test = pad_sequences(sentencepiece_sequences_test, maxlen=max_sequence_len)
len(sentencepiece_x_test)


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


# In[ ]:


ids_col = 'id'
prediction_col = 'sentiment'
ids = df_test[ids_col]

predictions_dict = {
    'sentencepiece_cnn': model1.predict(sentencepiece_x_test)[:, 1], # 0.92936
    'word_cnn': model2.predict(word_x_test)[:, 1] # 0.92520
}

for model_name, predictions in predictions_dict.items():
    print('generating submission for: ', model_name)
    submission_path = os.path.join(submission_dir, '{}_submission.csv'.format(model_name))
    df_submission = create_submission(ids, predictions, ids_col, prediction_col, submission_path)

# sanity check to make sure the size and the output of the submission makes sense
print(df_submission.shape)
df_submission.head()


# ## Summary

# We've looked at the performance of leveraging subword tokenization for our text classification task. Note that some other ideas that we did not try out are:
# 
# - Use [other word-level tokenizers](https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/). Another popular choice at the point of writing this documentation is [spacy's tokenizer](https://spacy.io/usage/spacy-101#annotations-token).
# - [Sentencepiece suggests](https://github.com/google/sentencepiece#trains-from-raw-sentences) that it can be trained on raw text without the need to perform language specific segmentation beforehand, e.g. using the spacy tokenizer on our raw text data before feeding it to sentencepiece to learn the subword vocabulary. We can conduct our own experiment on the task at hand to verify that claim. Sentencepiece also includes an [experiments page](https://github.com/google/sentencepiece/blob/master/doc/experiments.md) that documents some of the experiments they've conducted.

# # Reference

# - [Github: sentencepiece](https://github.com/google/sentencepiece)
# - [Blog: NLP - Four Ways to Tokenize Chinese Documents](https://medium.com/the-artificial-impostor/nlp-four-ways-to-tokenize-chinese-documents-f349eb6ba3c3)
