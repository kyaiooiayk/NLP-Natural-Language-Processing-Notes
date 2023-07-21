#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Word2Vect" data-toc-modified-id="Word2Vect-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Word2Vect</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Dataset" data-toc-modified-id="Dataset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Dataset</a></span></li><li><span><a href="#Cleaning" data-toc-modified-id="Cleaning-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Cleaning</a></span></li><li><span><a href="#Bigrams" data-toc-modified-id="Bigrams-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Bigrams</a></span></li><li><span><a href="#Most-Frequent-Words" data-toc-modified-id="Most-Frequent-Words-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Most Frequent Words</a></span></li><li><span><a href="#Training-the-model" data-toc-modified-id="Training-the-model-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Training the model</a></span></li><li><span><a href="#Exploring-the-model" data-toc-modified-id="Exploring-the-model-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Exploring the model</a></span></li><li><span><a href="#t-SNE" data-toc-modified-id="t-SNE-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>t-SNE</a></span></li><li><span><a href="#10-Most-similar-words-vs.-8-Random-words" data-toc-modified-id="10-Most-similar-words-vs.-8-Random-words-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>10 Most similar words vs. 8 Random words</a></span></li><li><span><a href="#10-Most-similar-words-vs.-10-Most-dissimilar" data-toc-modified-id="10-Most-similar-words-vs.-10-Most-dissimilar-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>10 Most similar words vs. 10 Most dissimilar</a></span></li><li><span><a href="#References" data-toc-modified-id="References-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>References</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Requirements</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Word embeddings with Word2Vect (introduced by Google) and t-SNE
# 
# </font>
# </div>

# # Word2Vect
# <hr style="border:2px solid black"> </hr>

# - The underlying assumption of Word2Vec is that two words sharing similar contexts also share a similar meaning 
# and consequently a similar vector representation from the model. For instance: "dog", "puppy" and "pup" are 
# often used in similar situations, with similar surrounding words like "good", "fluffy" or "cute", and according 
# to Word2Vec they will therefore share a similar vector representation.
# 
# - From this assumption, Word2Vec can be used to find out the relations between words in a dataset, compute the 
# similarity between them, or use the vector representation of those words as input for other applications such
# as text classification or clustering.

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[7]:


import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from spacy.cli.download import download
import itertools


# # Dataset
# <hr style="border:2px solid black"> </hr>

# In[2]:


"""
This dataset contains the characters, locations, episode details, and script lines for approximately 600
Simpsons episodes, dating back to 1989. It can be found here: 
https://www.kaggle.com/ambarish/fun-in-text-mining-with-simpsons/data (~25MB)
"""


# In[2]:


df = pd.read_csv('../DATASETS/simpsons_dataset.csv')
df.shape


# In[3]:


df.head()


# In[ ]:


"""
The missing values comes from the part of the script where something happens, but with no dialogue. 
For instance "(Springfield Elementary School: EXT. ELEMENTARY - SCHOOL PLAYGROUND - AFTERNOON)"
"""


# In[4]:


df.isnull().sum()


# In[5]:


# Removing the missing values
df = df.dropna().reset_index(drop=True)
df.isnull().sum()


# In[6]:


# Checking again the dataset is cleaned from null values
df.isnull().sum()


# # Cleaning
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
We are lemmatizing and removing the stopwords and non-alphabetic characters for each line of dialogue.
LEMMANTISATION: Compare to stemming, lemmatization is a robust, efficient and methodical way of combining 
grammatical variations to the root of a word. The “lemma” for a word is the base word from which it is derived. 
Intuitively, if we are making a bag-of-words, then “running,” “ran,” and “runs” should all count the same.
"""


# In[ ]:


"""
See this link on how to donwload the model
https://stackoverflow.com/questions/66087475/chatterbot-error-oserror-e941-cant-find-model-en
"""


# In[8]:


download(model="en_core_web_sm")
# Disabling Named Entity Recognition for speed
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) 


# In[27]:


# Removes non-alphabetic characters:
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])


# In[32]:


# Let us visualise the result. I need to make a copy otherwise it changes the object type!
brief_cleaning, brief_cleaning2 = itertools.tee(brief_cleaning)
pd.DataFrame(brief_cleaning2)


# In[25]:


def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long, the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)


# In[26]:


# Taking advantage of spaCy .pipe() attribute to speed-up the cleaning process:
t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_process=-1)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))


# In[29]:


# Put the results in a DataFrame to remove missing values and duplicates:
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape


# In[33]:


df_clean


# # Bigrams
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
We are using Gensim Phrases package to automatically detect common phrases (bigrams) from a list of sentences.
The main reason we do this is to catch words like "mr_burns" or "bart_simpson" !
"""


# In[34]:


# Phrases() takes a list of list of words as input:
sent = [row.split() for row in df_clean['clean']]


# In[35]:


# Creates the relevant phrases from the list of sentences
phrases = Phrases(sent, min_count=30, progress_per=10000)


# In[ ]:


"""
The goal of Phraser() is to cut down memory consumption of Phrases(), by discarding model state not strictly 
needed for the bigram detection task:
"""


# In[36]:


bigram = Phraser(phrases)


# In[37]:


# Transform the corpus based on the bigrams detected:
sentences = bigram[sent]


# In[39]:


list(sentences)


# # Most Frequent Words
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
Mainly a sanity check of the effectiveness of the lemmatization, removal of stopwords, and addition of bigrams.
"""


# In[29]:


word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)


# In[30]:


sorted(word_freq, key=word_freq.get, reverse=True)[:10]


# # Training the model
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
We use Gensim implementation of word2vec: https://radimrehurek.com/gensim/models/word2vec.html
3 distinctive steps for clarity and monitoring.

    Word2Vec():
        In this first step, I set up the parameters of the model one-by-one.
        I do not supply the parameter sentences, and therefore leave the model uninitialized, purposefully.

    .build_vocab():
        Here it builds the vocabulary from a sequence of sentences and thus initialized the model.
        With the loggings, I can follow the progress and even more important, the effect of min_count and 
        sample on the word corpus. I noticed that these two parameters, and in particular sample, have a 
        great influence over the performance of a model. Displaying both allows for a more accurate and 
        an easier management of their influence.

    .train():
        Finally, trains the model.
        The loggings here are mainly useful for monitoring, making sure that no threads are executed 
        instantaneously.
"""


# In[42]:


# Count the number of cores in a computer
cores = multiprocessing.cpu_count() 
print(cores)


# In[ ]:


"""
- min_count = int - Ignores all words with total absolute frequency lower than this - (2, 100)
- window = int - The maximum distance between the current and predicted word within a sentence. E.g. window words 
on the left and window words on the left of our target - (2, 10)
- size = int - Dimensionality of the feature vectors. - (50, 300)
- sample = float - The threshold for configuring which higher-frequency words are randomly downsampled. Highly 
  influencial. - (0, 1e-5)
- alpha = float - The initial learning rate - (0.01, 0.05)
- min_alpha = float - Learning rate will linearly drop to min_alpha as training progresses. To set it: 
  alpha - (min_alpha * epochs) ~ 0.00
- negative = int - If > 0, negative sampling will be used, the int for negative specifies how many "noise words" 
  should be drown. If set to 0, no negative sampling is used. - (5, 20)
- workers = int - Use these many worker threads to train the model (=faster training with multicore machines)
"""


# In[43]:


w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


# In[ ]:


"""
Word2Vec requires us to build the vocabulary table (simply digesting all the words and filtering out the unique 
words, and doing some basic counts on them). progress_per controls the update frequency.
"""


# In[46]:


t = time()
w2v_model.build_vocab(sentences, progress_per=5000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


"""
Parameters of the training:
    total_examples = int - Count of sentences;
    epochs = int - Number of iterations (epochs) over the corpus - [10, 20, 30]
"""


# In[47]:


t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


"""
As we do not plan to train the model any further, we are calling init_sims(), which will make the model
much more memory-efficient:
"""


# In[48]:


w2v_model.init_sims(replace=True)


# # Exploring the model
# <hr style="border:2px solid black"> </hr>

# In[41]:


"""
Here, we will ask our model to find the word most similar to some of the most iconic characters of the Simpsons!
we get what other characters (as Homer does not often refers to himself at the 3rd person) said along with "homer"
"""


# In[49]:


w2v_model.wv.most_similar(positive=["homer"])


# In[50]:


# Let's see what the bigram "homer_simpson" gives us by comparison:
w2v_model.wv.most_similar(positive=["homer_simpson"])


# In[51]:


# Here, we will see how similar are two words to each other
w2v_model.wv.similarity("maggie", 'baby')


# In[52]:


# Here, we ask our model to give us the word that does not belong to the list!
# Between Jimbo, Milhouse, and Kearney, who is the one who is not a bully?
w2v_model.wv.doesnt_match(['homer', 'milhouse', 'bart'])


# In[53]:


# Which word is to woman as homer is to marge?
w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)


# In[54]:


w2v_model.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3)


# # t-SNE
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
t-SNE is a non-linear dimensionality reduction algorithm that attempts to represent high-dimensional data 
and the underlying relationships between vectors in a lower-dimensional space.

Our goal in this section is to plot our 300 dimensions vectors into 2 dimensional graphs, and see if we can spot 
interesting patterns. For that we are going to use t-SNE implementation from scikit-learn.
To make the visualizations more relevant, we will look at the relationships between a query word (in **red**), 
its most similar words in the model (in **blue**), and other words from the vocabulary (in **green**).
"""


# In[61]:


def tsnescatterplot(model, word, list_names, n_components):
    """t-SNE scatter plot
    Plot in seaborn the results from the t-SNE dimensionality 
    reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """

    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    # print(n_components = min(n_samples, n_features))
    reduc = PCA(n_components=n_components).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0,
             perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - n_components, Y[:, 0].max() + n_components)
    plt.ylim(Y[:, 1].min() - n_components, Y[:, 1].max() + n_components)

    plt.title('t-SNE visualization for {}'.format(word.title()))


# # 10 Most similar words vs. 8 Random words
# <hr style="border:2px solid black"> </hr>

# In[59]:


"""
Let's compare where the vector representation of Homer, his 10 most similar words from the model, 
as well as 8 random ones, lies in a 2D graph:
"""


# In[62]:


tsnescatterplot(w2v_model, 'homer', ['dog', 'bird', 'ah', 'maude', 'bob', 'mel', 'apu', 'duff'], 19)


# # 10 Most similar words vs. 10 Most dissimilar
# <hr style="border:2px solid black"> </hr>

# In[ ]:


"""
This time, let's compare where the vector representation of Maggie and her 10 most similar words from the model
lies compare to the vector representation of the 10 most dissimilar words to Maggie:
"""


# In[63]:


tsnescatterplot(w2v_model, 'maggie', [i[0] for i in w2v_model.wv.most_similar(negative=["maggie"])], 19)


# # References
# <hr style="border:2px solid black"> </hr>

# - https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial/notebook

# # Requirements
# <hr style="border:2px solid black"> </hr>

# In[3]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv -m')


# In[ ]:




