#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Simple text encoding-decoding
# 
# </font>
# </div>

# # Imports
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# - The only not here is that I have put a place at the end as I'd like to encode that as well.
# - `list()` is used on the string to get each single character automatically.
# 
# </font>
# </div>

# In[6]:


chars = "abcdefghilmnopqrstuwvz "


# In[7]:


list(chars)


# In[11]:


# create a mapping from characters to integers
ch_to_nu = { ch:i for i,ch in enumerate(chars) }
no_to_ch = { i:ch for i,ch in enumerate(chars) }


# In[12]:


ch_to_nu 


# In[13]:


no_to_ch


# In[14]:


encode = lambda s: [ch_to_nu[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([no_to_ch[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=Yw1LKNCgwjj1
#     
# </font>
# </div>

# In[ ]:




