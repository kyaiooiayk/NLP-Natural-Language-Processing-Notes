#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Applying Zipf's law to text
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[12]:


import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 17, 8
rcParams['font.size'] = 20


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - **Zipf's law** states that the frequency of a token in a text is directly proportional to its rank or position in the sorted list. 
# - This law describes how tokens are distributed in languages: some tokens occur very frequently, some occur with intermediate frequency, and some tokens rarely occur. 
# 
# </font>
# </div>

# In[3]:


fd = FreqDist()
for text in gutenberg.fileids():
    for word in gutenberg.words(text):
        fd.update(word)

ranks = []
freqs = []
for rank, word in enumerate(fd):
    ranks.append(rank+1)
    freqs.append(fd[word])


# In[13]:


fig = plt.figure() 
ax = fig.add_subplot(111) 

plt.loglog(ranks, freqs)
plt.xlabel('frequency(f)', fontsize=14, fontweight='bold')
plt.ylabel('rank(r)', fontsize=14, fontweight='bold')
plt.grid(True)

ax.grid(which = "major", linestyle='-', linewidth='1.0', color='k')
ax.grid(which = "minor", linestyle='--', linewidth='0.25', color='k')
ax.tick_params(which = 'major', direction='in', length=10, width=2)
ax.tick_params(which = 'minor', direction='in', length=6, width=2)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.show()


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - Chopra, Deepti, Nisheeth Joshi, and Iti Mathur. Mastering natural language processing with python. Packt Publishing Ltd, 2016.
# - https://github.com/PacktPublishing/Mastering-Natural-Language-Processing-with-Python
#     
# </font>
# </div>

# In[ ]:




