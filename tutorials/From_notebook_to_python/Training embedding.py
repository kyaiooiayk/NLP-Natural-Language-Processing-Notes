#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Define-training-data" data-toc-modified-id="Define-training-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Define training data</a></span></li><li><span><a href="#Training-the-model" data-toc-modified-id="Training-the-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Training the model</a></span></li><li><span><a href="#Continuous-Bag-of-Words-(CBOW)" data-toc-modified-id="Continuous-Bag-of-Words-(CBOW)-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Continuous Bag of Words (CBOW)</a></span></li><li><span><a href="#SkipGram" data-toc-modified-id="SkipGram-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>SkipGram</a></span></li><li><span><a href="#Training-Your-Embedding-on-Wiki-Corpus" data-toc-modified-id="Training-Your-Embedding-on-Wiki-Corpus-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Training Your Embedding on Wiki Corpus</a></span></li><li><span><a href="#FastText" data-toc-modified-id="FastText-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>FastText</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# In[ ]:


"""
What? Training embedding

Word embeddings are an approach to representing text in NLP. In this notebook we will demonstrate how to train 
embeddings using Genism
"""


# # Imports

# In[32]:


from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
import os
import requests
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import time


# # Define training data

# In[ ]:


"""
Genism word2vec requires that a format of ‘list of lists’ be provided for training where every document 
contained in a list. Every list contains lists of tokens of that document.
"""


# In[45]:


corpus = [['dog','bites','man'], ["man", "bites" ,"dog"],["dog","eats","meat"],["man", "eats","food"]]
corpus


# # Training the model

# In[ ]:


"""
Two different learning models were introduced as part of the word2vec approach to learn the word embedding; 
they are:
    [1] Continuous Bag-of-Words, or CBOW model.
    [2] Continuous Skip-Gram Model
"""


# In[3]:


# Using CBOW Architecture for trainnig
model_cbow     = Word2Vec(corpus, min_count=1,sg=0) 
# Using skipGram Architecture for training 
model_skipgram = Word2Vec(corpus, min_count=1,sg=1)


# # Continuous Bag of Words (CBOW) 

# In[ ]:


"""
In CBOW, the primary task is to build a language model that correctly predicts the center word given the context 
words in which the center word appears.
"""


# In[15]:


#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.index_to_key)
print(words)

# Access vector for one word
print(model_cbow.wv['dog'])


# In[21]:


# Compute similarity 
print("Similarity between eats and bites:",model_cbow.wv.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_cbow.wv.similarity('eats', 'man'))


# From the above similarity scores we can conclude that eats is more similar to bites than man.

# In[22]:


#Most similarity
model_cbow.wv.most_similar('meat')


# In[6]:


# save model
model_cbow.save('model_cbow.bin')

# load model
new_model_cbow = Word2Vec.load('model_cbow.bin')
print(new_model_cbow)


# # SkipGram

# In[ ]:


"""
In skipgram, the task is to predict the context words from the center word.
"""


# In[27]:


#Summarize the loaded model
print(model_skipgram)

#Summarize vocabulary
words = list(model_skipgram.wv.index_to_key)
print(words)

#Acess vector for one word
print(model_skipgram.wv['dog'])


# In[28]:


#Compute similarity 
print("Similarity between eats and bites:",model_skipgram.wv.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_skipgram.wv.similarity('eats', 'man'))


# From the above similarity scores we can conclude that eats is more similar to bites than man.

# In[29]:


#Most similarity
model_skipgram.wv.most_similar('meat')


# In[10]:


# save model
model_skipgram.save('model_skipgram.bin')

# load model
new_model_skipgram = Word2Vec.load('model_skipgram.bin')
print(model_skipgram)


# # Training Your Embedding on Wiki Corpus

# In[ ]:


"""
The corpus download page : https://dumps.wikimedia.org/enwiki/20200120/
The entire wiki corpus as of 28/04/2020 is just over 16GB in size.
We will take a part of this corpus due to computation constraints and train our word2vec and fasttext embeddings.

The file size is 294MB so it can take a while to download.
"""


# In[31]:


os.makedirs('data/en', exist_ok= True)
file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p13159683p14324602.bz2"
file_id = "11804g0GcWnBIVDahjo5fQyc05nQLXGwF"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if not os.path.exists(file_name):
    download_file_from_google_drive(file_id, file_name)
else:
    print("file already exists, skipping download")

print(f"File at: {file_name}")


# In[34]:


#Preparing the Training data
wiki = WikiCorpus(file_name, dictionary={})
sentences = list(wiki.get_texts())


# In[ ]:


"""
1.   sg - Selecting the training algorithm: 1 for skip-gram else its 0 for CBOW. Default is CBOW.
2.   min_count-  Ignores all words with total frequency lower than this.<br>
There are many more hyperparamaeters whose list can be found in the official documentation [here.]
(https://radimrehurek.com/gensim/models/word2vec.html)
"""


# In[35]:


#CBOW
start = time.time()
word2vec_cbow = Word2Vec(sentences,min_count=10, sg=0)
end = time.time()

print("CBOW Model Training Complete.\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))


# In[38]:


#Summarize the loaded model
print(word2vec_cbow)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_cbow.wv.index_to_key)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_cbow.wv['film'])}")
print(word2vec_cbow.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_cbow.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_cbow.wv.similarity('film', 'tiger'))
print("-"*30)


# In[47]:


# save model
from gensim.models import Word2Vec, KeyedVectors   
word2vec_cbow.wv.save_word2vec_format('word2vec_cbow.bin', binary=True)

# load model
# new_modelword2vec_cbow = Word2Vec.load('word2vec_cbow.bin')
# print(word2vec_cbow)


# In[39]:


#SkipGram
start = time.time()
word2vec_skipgram = Word2Vec(sentences,min_count=10, sg=1)
end = time.time()

print("SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))


# In[40]:


#Summarize the loaded model
print(word2vec_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_skipgram.wv.index_to_key)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_skipgram.wv['film'])}")
print(word2vec_skipgram.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_skipgram.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_skipgram.wv.similarity('film', 'tiger'))
print("-"*30)


# In[51]:


# save model
word2vec_cbow.wv.save_word2vec_format('word2vec_sg.bin', binary=True)

# load model
# new_model_skipgram = Word2Vec.load('model_skipgram.bin')
# print(model_skipgram)


# # FastText

# In[ ]:


"""
When we have a large dataset, and when learning seems infeasible with the approaches described so far, 
fastText is a good option to use to set up a strong working baseline. However, there’s one concern to keep 
in mind when using fastText, as was the case with Word2vec embeddings: it uses pre-trained character n-gram
embeddings. Thus, when we save the trained model, it carries the entire character n-gram embeddings dictionary 
with it. This results in a bulky model and can result in engineering issues. For example, the model stored with 
the name “temp” in the above code snippet has a size close to 450 MB. However, fastText implementation also comes
with options to reduce the memory footprint of its classification models with minimal reduction in classification
performance.


Some of the most popular pre-trained embeddings are
    [1] Word2vec by Google
    [2] GloVe    by Stanford
    [3] fastText by Facebook
"""


# In[41]:


# CBOW
start = time.time()
fasttext_cbow = FastText(sentences, sg=0, min_count=10)
end = time.time()

print("FastText CBOW Model Training Complete\nTime taken for training is:{:.2f} hrs ".format(
    (end-start)/3600.0))


# In[42]:


# Summarize the loaded model
print(fasttext_cbow)
print("-"*30)

# Summarize vocabulary
words = list(fasttext_cbow.wv.index_to_key)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

# Acess vector for one word
print(f"Length of vector: {len(fasttext_cbow.wv['film'])}")
print(fasttext_cbow.wv['film'])
print("-"*30)

# Compute similarity
print("Similarity between film and drama:",
      fasttext_cbow.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",
      fasttext_cbow.wv.similarity('film', 'tiger'))
print("-"*30)


# In[43]:


#SkipGram
start = time.time()
fasttext_skipgram = FastText(sentences, sg=1, min_count=10)
end = time.time()

print("FastText SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0))


# In[44]:


#Summarize the loaded model
print(fasttext_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(fasttext_skipgram.wv.index_to_key)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(fasttext_skipgram.wv['film'])}")
print(fasttext_skipgram.wv['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",fasttext_skipgram.wv.similarity('film', 'drama'))
print("Similarity between film and tiger:",fasttext_skipgram.wv.similarity('film', 'tiger'))
print("-"*30)


# In[ ]:


"""
An interesting obeseravtion if you noticed is that CBOW trains faster than SkipGram in both cases.
"""


# # References

# - https://github.com/practical-nlp/practical-nlp/blob/master/Ch3/06_Training_embeddings_using_gensim.ipynb
# - Harshit Surana, Practical Natural Language Processing
# 

# In[ ]:




