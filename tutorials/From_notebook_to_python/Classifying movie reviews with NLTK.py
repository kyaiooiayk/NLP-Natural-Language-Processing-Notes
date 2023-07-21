#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Project's-goal" data-toc-modified-id="Project's-goal-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Project's goal</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-warning">
# <font color=black>
# 
# **What?** Classifying movie reviews with NLTK
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>

# In[2]:


import nltk
from nltk.corpus import movie_reviews


# # Project's goal
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Using these corpora, we can build classifiers that will automatically tag new documents with appropriate category labels. 
# - First, we construct a list of docu- ments, labeled with the appropriate categories. 
# - For this example, we’ve chosen the Movie Reviews Corpus, which categorizes each review as positive or negative.
# 
# </font>
# </div>

# In[4]:


movie_reviews.categories()


# In[5]:


documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))


# In[6]:


len(documents)


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Next, we define a feature extractor for documents, so the classifier will know which aspects of the data it should pay attention to (see Example 6-2). 
# - For document topic identification, we can define a feature for each word, indicating whether the document contains that word. 
# - To limit the number of features that the classifier needs to process, we begin by constructing a list of the 2,000 most frequent words in the overall corpus. 
# - We can then define a feature extractor that simply checks whether each of these words is present in a given document.
# 
# </font>
# </div>

# In[8]:


all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words()) 
word_features = list(all_words.keys())[:2000]


# In[9]:


word_features 


# In[10]:


def document_features(document):
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words) 
    return features


# In[11]:


document_features(movie_reviews.words('pos/cv957_8737.txt'))


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Now that we’ve defined our feature extractor, we can use it to train a classifier to label new movie reviews. 
# - To check how reliable the resulting classifier is, we compute its accuracy on the test set . 
# - And once again, we can use show_most_infor mative_features() to find out which features the classifier found to be most informative
# 
# </font>
# </div>

# In[13]:


featuresets = [(document_features(d), c) for (d,c) in documents] 
train_set, test_set = featuresets[100:], featuresets[:100] 
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[14]:


nltk.classify.accuracy(classifier, test_set)


# In[15]:


classifier.show_most_informative_features(5)


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Apparently in this corpus, a review that mentions "schumacher" is almost 8 times more likely to be negative than positive, while a review that mentions "recongizes" is about 8 times more likely to be positive.
# 
# </font>
# </div>

# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - Bird, Steven, Ewan Klein, and Edward Loper. Natural language processing with Python: analyzing text with the natural language toolkit. " O'Reilly Media, Inc.", 2009.
# - https://github.com/Sturzgefahr/Natural-Language-Processing-with-Python-Analyzing-Text-with-the-Natural-Language-Toolkit
# 
# </font>
# </div>

# In[ ]:




