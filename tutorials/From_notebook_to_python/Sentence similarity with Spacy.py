#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Download-corpus" data-toc-modified-id="Download-corpus-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Download corpus</a></span></li><li><span><a href="#Create-and-compare-sentence-similarity" data-toc-modified-id="Create-and-compare-sentence-similarity-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Create and compare sentence similarity</a></span></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Sentence similarity with Spacy
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[1]:


import spacy


# # Download corpus
# <hr style = "border:2px solid black" ></hr>

# In[2]:


# Download and extract the corpus
#!python -m spacy download en_core_web_lg


# In[ ]:


nlp = spacy.load("en_core_web_lg")
#nlp = spacy.load("en_core_web_md")


# # Create and compare sentence similarity
# <hr style = "border:2px solid black" ></hr>

# In[5]:


doc1 = nlp(u'the person wear red T-shirt')
doc2 = nlp(u'this person is walking')
doc3 = nlp(u'the boy wear red T-shirt')


print(doc1.similarity(doc2))
print(doc1.similarity(doc3))
print(doc2.similarity(doc3))


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://stackoverflow.com/questions/65199011/is-there-a-way-to-check-similarity-between-two-full-sentences-in-python
#     
# </font>
# </div>

# In[ ]:




