#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Intro-on-Doc2Vec" data-toc-modified-id="Intro-on-Doc2Vec-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Intro on Doc2Vec</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Doc2Vec-by-averaging" data-toc-modified-id="Doc2Vec-by-averaging-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Doc2Vec by averaging</a></span></li><li><span><a href="#Training-Doc2Vec" data-toc-modified-id="Training-Doc2Vec-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Training Doc2Vec</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Doc2Vec embedding
# 
# </font>
# </div>

# # Intro on Doc2Vec
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - In the Doc2vec embedding scheme, we learn a direct representation for the entire document (sentence/paragraph) rather than each word. 
# - Just as we used word and character embeddings as features for performing text classification, we can also use Doc2vec as a feature representation mechanism. 
# - Doc2vec allows us to directly learn the representations for texts of arbitrary lengths (phrases, sentences, paragraphs and documents), by considering the context of words  in the text into account. 
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[1]:


import spacy
import warnings
warnings.filterwarnings('ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pprint import pprint
import nltk
nltk.download('punkt')


# In[2]:


# downloading en_core_web_sm, assuming spacy is already installed
get_ipython().system('python -m spacy download en_core_web_sm')


# In[3]:


#here nlp object refers to the 'en_core_web_sm' language model instance.
nlp = spacy.load("en_core_web_sm") 


# # Doc2Vec by averaging
# <hr style="border:2px solid black"> </hr>

# In[4]:


# Assume each sentence in documents corresponds to a separate document.
documents = ["Dog bites man.", "Man bites dog.",
             "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".", "") for doc in documents]
processed_docs
print("Document After Pre-Processing:", processed_docs)


# In[5]:


# Iterate over each document and initiate an nlp instance.
for doc in processed_docs:
    # creating a spacy "Doc" object which is a container for accessing linguistic annotations.
    doc_nlp = nlp(doc)

    print("-"*30)
    # this gives the average vector of each document
    print("Average Vector of '{}'\n".format(doc), doc_nlp.vector)
    for token in doc_nlp:
        print()
        # this gives the text of each word in the doc and their respective vectors.
        print(token.text, token.vector)


# # Training Doc2Vec
# <hr style="border:2px solid black"> </hr>

# In[6]:


data = ["dog bites man",
        "man bites dog",
        "dog eats meat",
        "man eats food"]

tagged_data = [TaggedDocument(words=word_tokenize(word.lower()), tags=[
                              str(i)]) for i, word in enumerate(data)]


# In[7]:


tagged_data


# In[10]:


# dbow
#model_dbow = Doc2Vec(tagged_data, vector_size=20, min_count=1, epochs=2, dm=0)
model_dbow = Doc2Vec(tagged_data, min_count=1 , dm=0)


# In[11]:


# feature vector of man eats food
print(model_dbow.infer_vector(['man','eats','food'])) 


# In[12]:


# top 5 most simlar words
model_dbow.wv.most_similar("man", topn=5)


# In[13]:


model_dbow.wv.n_similarity(["dog"],["man"])


# In[15]:


# dm
#model_dm = Doc2Vec(tagged_data, min_count=1, vector_size=20, epochs=2, dm=1)
model_dm = Doc2Vec(tagged_data, min_count=1, dm=1)

print("Inference Vector of man eats food\n ",
      model_dm.infer_vector(['man', 'eats', 'food']))

print("Most similar words to man in our corpus\n",
      model_dm.wv.most_similar("man", topn=5))
print("Similarity between man and dog: ",
      model_dm.wv.n_similarity(["dog"], ["man"]))


# In[16]:


# What happens when we compare between words which are not in the vocabulary?
model_dm.wv.n_similarity(['covid'],['man'])


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/practical-nlp/practical-nlp/blob/master/Ch3/07_DocVectors_using_averaging_Via_spacy.ipynb
# - https://github.com/practical-nlp/practical-nlp/blob/master/Ch3/08_Training_Dov2Vec_using_Gensim.ipynb
# </font>
# </div>

# In[ ]:




