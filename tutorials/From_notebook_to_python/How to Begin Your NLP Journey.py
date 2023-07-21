#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Define-your-text" data-toc-modified-id="Define-your-text-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Define your text</a></span></li><li><span><a href="#Tokenization" data-toc-modified-id="Tokenization-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Tokenization</a></span></li><li><span><a href="#Lowercase-&amp;-punctuation" data-toc-modified-id="Lowercase-&amp;-punctuation-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Lowercase &amp; punctuation</a></span></li><li><span><a href="#Stopwords-removal" data-toc-modified-id="Stopwords-removal-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Stopwords removal</a></span></li><li><span><a href="#Regular-expressions" data-toc-modified-id="Regular-expressions-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Regular expressions</a></span></li><li><span><a href="#References" data-toc-modified-id="References-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** How to Begin Your NLP Journey
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[36]:


import langdetect
from langdetect import detect_langs
import nltk
import re
from nltk import FreqDist
from nltk.tokenize import word_tokenize


# # Define your text
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - We're going to take a quote by Stephen Hawking.
# - Using the langdetect library, we can check its language, and find out the probability of being written in that language.
# 
# </font>
# </div>

# In[6]:


text = "Artificial Intelligence (AI) is likely to be either the best or the worst thing to happen to humanity."


# In[9]:


print(detect_langs(text))


# # Tokenization
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Tokenization is the process of segmenting running text into sentences and words. 
# - In essence, it’s the task of cutting a text into pieces called tokens. 
# - We use the NLTK library to perform this task:
# 
# </font>
# </div>

# In[ ]:


nltk.download('punkt')
tokenized_word = word_tokenize(text)
print(tokenized_word)


# In[13]:


# Now we can calculate a measure related to the lexical richness of the text:
#This shows that the number of distinct words is 85,7% of the total number of words.
len(set(tokenized_word)) / len(tokenized_word)


# # Lowercase & punctuation
# <hr style="border:2px solid black"> </hr>

# In[ ]:


# Now let’s lowercase the text to standardize characters and for future stopwords removal:


# In[24]:


tk_low_np = [w.lower() for w in tokenized_word]
print(tk_low)


# In[ ]:


# Next, we remove non-alphanumerical characters:


# In[25]:


fdist = FreqDist(tk_low_np)
fdist.plot(title = "Word frequency distribution", cumulative = True)


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - We can see that the words “to” and “the” appear most often, but they don’t really add information to the text.
# - They are what’s known as **stopwords**.
# 
# </font>
# </div>

# # Stopwords removal
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - This process includes getting rid of common language articles, pronouns and prepositions such as “and”, “the” or “to” in English. 
# - In this process some very common words that appear to provide little or no value to the NLP objective are filtered and excluded from the text to be processed, hence removing widespread and frequent terms that are not informative about the corresponding text.
# 
# </font>
# </div>

# In[29]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print(stop_words)


# In[ ]:


# Now, let’s clean our text from these stopwords:


# In[31]:


filtered_text = []
for w in tk_low_np:
    if w not in stop_words:
        filtered_text.append(w)
print(filtered_text)


# In[32]:


fdist = FreqDist(filtered_text)
fdist.plot(title = "Word frequency distribution", cumulative = True)


# # Regular expressions
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Regular expressions (called REs, or RegExes) are a tiny, highly specialized programming language embedded inside Python and made available through the re module. 
# 
# </font>
# </div>

# In[35]:


# For example, let’s search for words ending with “st”:
[w for w in filtered_text if re.search('st$', w)]


# In[39]:


# Or count the number of vowels in the first word (“artificial”)
len(re.findall(r'[aeiou]', filtered_text[0]))


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/03/begin-nlp-journey.html
# 
# </font>
# </div>

# In[ ]:




