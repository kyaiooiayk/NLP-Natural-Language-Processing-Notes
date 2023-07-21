#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Tokenisation" data-toc-modified-id="Tokenisation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Tokenisation</a></span></li><li><span><a href="#Unigrams,-Bigrams,-Trigrams,-...,-N-grams" data-toc-modified-id="Unigrams,-Bigrams,-Trigrams,-...,-N-grams-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Unigrams, Bigrams, Trigrams, ..., N-grams</a></span></li><li><span><a href="#Lemmas-and-stems" data-toc-modified-id="Lemmas-and-stems-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Lemmas and stems</a></span></li><li><span><a href="#Categorizing-Words:-POS-Tagging" data-toc-modified-id="Categorizing-Words:-POS-Tagging-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Categorizing Words: POS Tagging</a></span></li><li><span><a href="#Categorizing-Spans:-Chunking-and-Named-Entity-Recognition" data-toc-modified-id="Categorizing-Spans:-Chunking-and-Named-Entity-Recognition-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Categorizing Spans: Chunking and Named Entity Recognition</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Introduction to SpaCy
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[3]:


import spacy
from nltk.tokenize import TweetTokenizer
import warnings
warnings.filterwarnings("ignore")


# # Tokenisation
# <hr style="border:2px solid black"> </hr>

# In[4]:


nlp = spacy.load('en_core_web_sm')
text = "Mary, don’t slap the green witch" 
print([str(token) for token in nlp(text.lower())])


# In[10]:


tweet = u"Snow White and the Seven Degrees #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))


# # Unigrams, Bigrams, Trigrams, ..., N-grams
# <hr style="border:2px solid black"> </hr>

# In[14]:


def n_grams(text, n):
    """Custom-made n-grams constructor
    takes tokens or text, returns a list of n-grams
    """
    return [text[i:i+n] for i in range(len(text)-n+1)]


# In[15]:


cleaned = ['mary', ',', "n't", 'slap', 'green', 'witch', '.'] 
print(n_grams(cleaned, 3))


# # Lemmas and stems
# <hr style="border:2px solid black"> </hr>

# In[17]:


doc = nlp(u"he was running late for school") 
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))


# # Categorizing Words: POS Tagging
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - We can extend the concept of labeling from documents to individual words or tokens.
# - A common example ofcategorizing words is part-of-speech (POS) tagging.
# 
# </font>
# </div>

# In[ ]:


doc = nlp(u"Mary slapped the green witch.") 
for token in doc:
    print('{} - {}'.format(token, token.pos_))


# # Categorizing Spans: Chunking and Named Entity Recognition
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Often, we need to label a span of text; that is, a contiguous multitoken boundary. 
# - For example, consider the sentence, “Mary slapped the green witch.” 
# - We might want to identify the noun phrases (NP) and verb phrases (VP) in it, as shown here: This is called **chunking** or **shallow parsing**.
# - Shallow parsing aims to derive higher-order units composed of the grammatical atoms, like nouns, verbs, adjectives, and so on. NP stands for Noun Phrase.
# 
# </font>
# </div>

# In[20]:


doc = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print('{} - {}'.format(chunk, chunk.label_))


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - Rao, Delip, and Brian McMahan. Natural language processing with PyTorch: build intelligent language applications using deep learning. " O'Reilly Media, Inc.", 2019.
# - https://github.com/joosthub/PyTorchNLPBook
#     
# </font>
# </div>

# In[ ]:




