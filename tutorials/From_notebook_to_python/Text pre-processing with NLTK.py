#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Text pre-processing with NLTK

NLTK is a standard python library for natural language processing and computational linguistics.

Reference: https://www.mygreatlearning.com/blog/nltk-tutorial-with-python/?highlight=nlp
"""


# # Import modules

# In[5]:


import nltk


# # Download nltk dataset

# ![image.png](attachment:image.png)

# In[2]:


nltk.download()


# # Accessing a datase in NLTK

# In[ ]:


"""
A corpus is essentially a collection of sentences which serves as an input. For further processing a corpus 
is broken down into smaller pieces.
"""


# In[6]:


from nltk.corpus import movie_reviews


# In[8]:


movie_reviews.words()


# In[9]:


len(movie_reviews.words())


# # Tokenisation

# In[ ]:


"""
Word tokenization is the process of breaking a sentence into words. 
"""


# In[10]:


from nltk.tokenize import word_tokenize


# In[11]:


data = "I pledge to be a data scientist one day!"


# In[12]:


word_tokenize(data)


# In[ ]:


"""
Sentence tokenization is the process of breaking a corpus into sentence level tokens. It’s essentially used 
when the corps consists of multiple paragraphs. Each paragraph is broken down into sentences
"""


# In[13]:


from nltk.tokenize import sent_tokenize


# In[14]:


sentence = "Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, and pies.The most commonly used cake ingredients include flour, sugar, eggs, butter or oil or margarine, a liquid, and leavening agents, such as baking soda or baking powder. Common additional ingredients and flavourings include dried, candied, or fresh fruit, nuts, cocoa, and extracts such as vanilla, with numerous substitutions for the primary ingredients.Cakes can also be filled with fruit preserves, nuts or dessert sauces (like pastry cream), iced with buttercream or other icings, and decorated with marzipan, piped borders, or candied fruit."


# In[15]:


sent_tokenize(sentence)


# In[16]:


len(sent_tokenize(sentence))


# # Punctuation Removal

# In[17]:


from nltk.tokenize import RegexpTokenizer


# In[18]:


tokenizer = RegexpTokenizer(r'\w+')


# In[19]:


tokenizer.tokenize("WoW! I am excited to learn data science!")


# # Stop words removal

# In[ ]:


"""
Stop words are words which occur frequently in a corpus. e.g a, an, the, in. Frequently occurring words are 
removed from the corpus for the sake of text-normalization.
"""


# In[20]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
to_be_removed = set(stopwords.words('english')) 
para="Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked. In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations  that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards,  and pies." 
tokenized_para=word_tokenize(para) 
print(tokenized_para) 
modified_token_list=[word for word in tokenized_para if not word in to_be_removed] 
print(modified_token_list)


# # Stemming

# In[ ]:


"""
It is reduction of inflection from words. Words with same origin will get reduced to a form which may or may not 
be a word.
NLTK has different stemmers which implement different methodologies.
"""


# ## Porter Stemmer

# In[21]:


from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
content = "Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations  that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, and pies."
tk_content=word_tokenize(content) 
stemmed_words = [stemmer.stem(i) for i in tk_content]  
print(stemmed_words)


# ## Lancaster Stemmer

# In[22]:


from nltk.stem import LancasterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
tk_content=word_tokenize(content) 
stemmed_words = [stemmer.stem(i) for i in tk_content] 
print(stemmed_words)


# # Lemmantisation

# In[ ]:


"""
It is another process of reducing inflection from words. The way its different from stemming is that it reduces
words to their origins which have actual meaning. Stemming sometimes generates words which are not even words.
"""


# In[25]:


import nltk 
from nltk.stem import WordNetLemmatizer 
lemmatizer=WordNetLemmatizer() 
tk_content=word_tokenize(content) 
lemmatized_words = [lemmatizer.lemmatize(i) for i in tk_content]  
print(lemmatized_words)


# # Chunking

# In[ ]:


"""
Chunking also known as shallow parsing, is practically a method in NLP applied to POS tagged data to gain 
further insights from it. It is done by grouping certain words on the basis of a pre-defined rule. The 
text is then parsed according to the rule to group data for phrase creation
"""


# In[32]:


import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
content = "Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked."
tokenized_text = nltk.word_tokenize(content) 
tagged_token = nltk.pos_tag(tokenized_text) 
grammer = "NP: {<DT>?<JJ>*<NN>}" 
phrases = nltk.RegexpParser(grammer) 
result = phrases.parse(tagged_token)
print(result)


# # Bag of words

# In[ ]:


"""
Bag of words is a simplistic model which gives information about the contents of a corpus in terms of number 
of occurrences of words. It ignores the grammar and context of the documents and is a mapping of words to their
counts in the corpus.
"""


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd  
content = """Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked. 
In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, and pies."""
count_vectorizer = CountVectorizer()  
bag_of_words = count_vectorizer.fit_transform(content.splitlines())
pd.DataFrame(bag_of_words.toarray(), columns = count_vectorizer.get_feature_names())


# # Synonymus using wordnet

# In[ ]:


"""
Wordnet is a cool corpus in NLTK which can be used to generate synonyms antonyms of words.
"""


# In[44]:


from nltk.corpus import wordnet 
syns = wordnet.synsets("dog")     
print(syns[0].name())     
print(syns[0].lemmas()[0].name())    
print(syns[0].definition())     
print(syns[0].examples())


# # Frequency distribution of words

# In[45]:


import nltk 
import matplotlib.pyplot as plt 
content = """Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked. 
In their oldest forms, cakes were modifications of bread""" 
words = nltk.tokenize.word_tokenize(content) 
fd = nltk.FreqDist(words) 
fd.plot()

