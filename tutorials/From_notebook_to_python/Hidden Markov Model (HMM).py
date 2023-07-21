#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#High-level-view-of-methods-available-in-NLP" data-toc-modified-id="High-level-view-of-methods-available-in-NLP-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>High level view of methods available in NLP</a></span></li><li><span><a href="#What-is-a-hidden-Markov-model?" data-toc-modified-id="What-is-a-hidden-Markov-model?-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>What is a hidden Markov model?</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Implementation" data-toc-modified-id="Implementation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Implementation</a></span></li><li><span><a href="#References" data-toc-modified-id="References-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Hidden Markov Model (HMM)
# 
# </font>
# </div>

# # High level view of methods available in NLP
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The different approaches used to solve NLP problems commonly fall into three categories:
#     - **Heuristics**: dictionaries and thesauruses
#     - **Machine learning**: Naive Bayes, SVM, hidden Markov model, conditional random fields
#     - **Deep Learning**: RNNs, LSTMs, GRUs, CNNs, transformer, autoencoder
# 
# </font>
# </div>

# # What is a hidden Markov model?
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - The hidden Markov model (HMM) is a statistical model that assumes there is an underlying, unobservable process  with hidden states that generates the data—i.e., we can only observe the data once it is generated. 
# - HMMs also  make the Markov assumption, which means that each hidden state is dependent on the previous state(s).
# - Consider the NLP task of part-of-speech (POS) tagging, which deals with assigning part-of-speech tags to sentences.
# - Parts of speech like JJ (adjective) and NN (noun) are hidden states, while the sentence “natural language processing (nlp)…” is directly observed.”
# 
# </font>
# </div>

# ![image.png](attachment:image.png)

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[2]:


import nltk
from nltk.util import unique_list


# # Implementation
# <hr style="border:2px solid black"> </hr>

# In[14]:


corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:700]
print(len(corpus))

tag_set = unique_list(tag for sent in corpus for (word, tag) in sent)
print(len(tag_set))

symbols = unique_list(word for sent in corpus for (word, tag) in sent)
print(len(symbols))
print(len(tag_set))

symbols = unique_list(word for sent in corpus for (word, tag) in sent)
print(len(symbols))


trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
train_corpus = []
test_corpus = []
for i in range(len(corpus)):
    if i % 10:
        train_corpus += [corpus[i]]
    else:
        test_corpus += [corpus[i]]

print(len(train_corpus))
print(len(test_corpus))

print("111")

# def train_and_test(est):
hmm = trainer.train_supervised(train_corpus)
print('%.2f%%' % (100 * hmm.evaluate(test_corpus)))


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - Chopra, Deepti, Nisheeth Joshi, and Iti Mathur. Mastering natural language processing with python. Packt Publishing Ltd, 2016.
# - https://tedboy.github.io/nlps/generated/generated/nltk.tag.HiddenMarkovModelTrainer.train_supervised.html
# - https://github.com/PacktPublishing/Mastering-Natural-Language-Processing-with-Python
#     
# </font>
# </div>

# In[ ]:




