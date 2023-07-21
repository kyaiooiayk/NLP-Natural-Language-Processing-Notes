#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Working with NLTK package

Reference: Bird, Steven, Ewan Klein, and Edward Loper. Natural language processing with Python: analyzing text 
           with the natural language toolkit. " O'Reilly Media, Inc.", 2009.
https://github.com/Sturzgefahr/Natural-Language-Processing-with-Python-Analyzing-Text-with-the-Natural-Language-Toolkit
"""


# # Import modules

# In[21]:


from nltk.book import *
import nltk


# In[2]:


text1


# # Searching text

# In[ ]:


"""
A concordance view shows us every occurrence of a given word, together with some context.
"""


# In[3]:


text1.concordance("monstrous")


# In[ ]:


"""
What other words appear in a similar range of contexts? We can find out by appending the term similar
to the name of the text in question, then inserting the relevant word in parentheses:
"""


# In[4]:


text1.similar("monstrous")


# In[ ]:


"""
The term common_contexts allows us to examine just the contexts that are shared by two or more
words, such as monstrous and very.
"""


# In[5]:


text2.common_contexts(["monstrous", "very"])


# In[ ]:


"""
However, we can also determine the location of a word in the text: how many words from the beginning it appears. 
This positional information can be displayed using a dispersion plot. Each stripe represents an instance of a 
word, and each row represents the entire text. Dispersion plot can ve ysed to investigate changes in language 
use over time.
"""


# In[6]:


text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# In[ ]:


"""
Now, just for fun, let’s try generating some random text in the various styles we have just seen. To do this
we type the name of the text followed by the term generate. Although the text is random, it reuses common words
and phrases from the source text and gives us a sense of its style and content.
"""


# In[7]:


text3.generate()


# # Counting Vocabulary

# In[9]:


len(text3)


# In[13]:


sorted(set(text3))


# In[14]:


len(set(text3))


# In[ ]:


"""
Now, let’s calculate a measure of the lexical richness of the text. The next example shows us that each word is
used 16 times on average 
"""


# In[15]:


len(text3) / len(set(text3))


# # Collocations and Bigrams

# In[ ]:


"""
A collocation is a sequence of words that occur together unusually often. Thus red wine is a collocation, 
whereas the wine is not. A characteristic of collocations is that they are resistant to substitution with 
words that have similar senses; for example, maroon wine sounds very odd.
"""


# In[17]:


list(bigrams(['more', 'is', 'said', 'than', 'done']))


# In[ ]:


"""
Now, collocations are essentially just frequent bigrams, exceptthat we want to pay more attention to the 
cases that involve rare words. In particular, we want to find bigrams that occur more often than we would 
expect based on the fre- quency of individual words. The collocations() function does this for us
"""


# In[18]:


text4.collocations()


# # OPS tagger

# In[ ]:


"""
A part-of-speech tagger, or POS tagger, processes a sequence of words, and attaches a part of speech tag to each word
CC = coordinating conjunction
RB = adverbs
IN = preposition 
NN = noun
JJ = adjective
"""


# In[22]:


text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)


# In[ ]:




