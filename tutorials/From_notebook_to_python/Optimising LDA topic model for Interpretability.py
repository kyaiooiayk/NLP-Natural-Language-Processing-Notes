#!/usr/bin/env python
# coding: utf-8

# # Introduction
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Optimising LDA topic model for Interpretability
# 
# </font>
# </div>

# # LDA
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - **Latent Dirichlet Allocation** (LDA) assumes each document consists of a combination of topics, and each topic consists of a combination of words.
# - It then approximates probability distributions of topics in a given document and of words in a given topic. For example:
#    - Document 1: Topic1 = 0.33, Topic2 = 0.33, Topic3 = 0.33
#    - Topic 1: Product = 0.39, Payment = 0.32, Store = 0.29
# - LDA is a type of Bayesian Inference Model. It assumes that the topics are generated before documents, and infer topics that could have generated the corupus of documents (a review = a document). 
# - The dimensionality K of Dirichlet distribution (aka # of topics) is assumed to be known and fixed and needs to be provided. This is generally not extremely easy to do and lack if intuitiveness. We'll try to address this now. 
# 
# </font>
# </div>

# # Imports
# <hr style="border:2px solid black"> </hr>
# 

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[2]:


import spacy
import nltk
import re
import string
import pandas as pd
import numpy as np
from stop_word_list import *
from clean_text import *
import gensim
from gensim import corpora
# Change in import see: https://stackoverflow.com/questions/66759852/no-module-named-pyldavis
import pyLDAvis.gensim_models #as gensimvis
import matplotlib.pyplot as plt
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load dataset
# <hr style="border:2px solid black"> </hr>

# - Amazon Office Product reviews.
# - See references to download the dataset.

# In[3]:


# Load the data
data = []
for line in open('reviews_Office_Products_5.json', 'r'):
    data.append(json.loads(line))


# In[4]:


df = pd.DataFrame(data)


# In[5]:


df.head()


# In[6]:


# Extract only reviews text
reviews = pd.DataFrame(df.reviewText)


# In[7]:


reviews


# # Clean dataset
# <hr style="border:2px solid black"> </hr>

# In[8]:


# Cleaning up takes some time, so I have reduced the dataset down to 1000
reviews = reviews[:1000]


# In[9]:


reviews


# In[10]:


clean_reviews = clean_all(reviews, 'reviewText')


# In[11]:


clean_reviews.head(10)


# # Form Bigrams & Trigrams
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - We want to identify bigrams and trigrams so we can concatenate them and consider them as one word. Bigrams are phrases containing 2 words e.g. ‘social media’, where ‘social’ and ‘media’ are more likely to co-occur rather than appear separately. Likewise, trigrams are phrases containing 3 words that more likely co-occur e.g. ‘Proctor and Gamble’.
# - We use Pointwise Mutual Information score to identify significant bigrams and trigrams to concatenate. We also filter bigrams or trigrams with the filter (noun/adj, noun), (noun/adj,all types,noun/adj) because these are common structures pointing out noun-type n-grams.
# - This helps the LDA model better cluster topics.
# - Moreover, we'd like to try to avoid to rank only single words. When you are asked, about most common topic, people are likely to reply with a short sentence rather than by keywords.
# 
# </font>
# </div>

# In[12]:


[comment.split() for comment in clean_reviews.reviewText]


# In[13]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents(
    [comment.split() for comment in clean_reviews.reviewText])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(50)
bigram_scores = finder.score_ngrams(bigram_measures.pmi)


# In[14]:


bigram_scores[:10]


# In[15]:


trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = nltk.collocations.TrigramCollocationFinder.from_documents(
    [comment.split() for comment in clean_reviews.reviewText])
# Filter only those that occur at least 50 times
finder.apply_freq_filter(50)
trigram_scores = finder.score_ngrams(trigram_measures.pmi)


# In[16]:


trigram_scores[:10]


# In[17]:


# Create pandas dataframe
bigram_pmi = pd.DataFrame(bigram_scores)
bigram_pmi.columns = ['bigram', 'pmi']
bigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)


# In[18]:


bigram_pmi.head(3)


# In[19]:


# Create pandas dataframe
trigram_pmi = pd.DataFrame(trigram_scores)
trigram_pmi.columns = ['trigram', 'pmi']
trigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)


# In[20]:


trigram_pmi.head(3)


# In[21]:


# Filter for bigrams with only noun-type structures
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop_word_list or bigram[1] in stop_word_list:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True


# In[22]:


# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ', 'NN']:
        return False
    if trigram[0] in stop_word_list or trigram[-1] in stop_word_list or trigram[1] in stop_word_list:
        return False
    if 'n' in trigram or 't' in trigram:
        return False
    if 'PRON' in trigram:
        return False
    return True


# In[23]:


# Can set pmi threshold to whatever makes sense - eyeball through and select threshold where n-grams stop making sense
# choose top 500 ngrams in this case ranked by PMI that have noun like structures
filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram:
                                              bigram_filter(bigram['bigram'])
                                              and bigram.pmi > 5, axis=1)][:500]

filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram:
                                                 trigram_filter(
                                                     trigram['trigram'])
                                                 and trigram.pmi > 5, axis=1)][:500]


bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(
    x[0]) > 2 or len(x[1]) > 2]
trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(
    x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]


# In[24]:


# examples of bigrams
bigrams[:10]


# In[25]:


# examples of trigrams
trigrams[:10]


# In[26]:


# Concatenate n-grams
def replace_ngram(x):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x


# In[41]:


clean_reviews.iloc[-1].reviewText


# In[28]:


reviews_w_ngrams = clean_reviews.copy()


# In[29]:


reviews_w_ngrams.reviewText = reviews_w_ngrams.reviewText.map(
    lambda x: replace_ngram(x))


# In[30]:


reviews_w_ngrams.head(2)


# In[31]:


reviews_w_ngrams.iloc[0].reviewText


# In[33]:


# tokenize reviews + remove stop words + remove names + remove words with less than 2 characters
reviews_w_ngrams = reviews_w_ngrams.reviewText.map(lambda x: [word for word in x.split()
                                                              if word not in stop_word_list
                                                              and word not in english_names
                                                              and len(word) > 2])


# In[34]:


reviews_w_ngrams.iloc[0]


# In[35]:


reviews_w_ngrams.head()


# In[ ]:


stop


# # Filter for only nouns
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
# 
# - Nouns are most likely indicators of a topic. For example, for the sentence ‘The store is nice’, we know the sentence is talking about ‘store’.
# - The other words in the sentence provide more context and explanation about the topic (‘store’) itself.
# - Therefore, filtering for the noun cleans the text for words that are more interpretable in the topic model.
# 
# </font>
# </div>

# In[36]:


# Filter for only nouns
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered


# In[37]:


final_reviews = reviews_w_ngrams.map(noun_only)


# In[60]:


final_reviews


# # LDA Model
# <hr style="border:2px solid black"> </hr>

# In[61]:


dictionary = corpora.Dictionary(final_reviews)


# In[62]:


doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_reviews]


# # Optimize # of k topics
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - LDA requires that we specify the number of topics that exists in a corpus of text.
# - There are several common measures that can be optimised, such as predictive likelihood, perplexity, and coherence.
# - Much literature has indicated that maximizing coherence, particularly a measure named Cv (https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf), leads to better human interpretability.
# - This measure assesses the interpretability of topics given the set of words in generated topics. Therefore, we will optimise this measure.
# - Since `eval_only` calculates perplexity metric, we can set it to `None` to save time, as we will use a different metric called Cv.
#     
# </font>
# </div>

# In[63]:


coherence = []
# 5 to 10 topic
for k in range(5, 10):
    print('Round: '+str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=40,
                   iterations=200, chunksize=10000, eval_every=None)

    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=final_reviews,
                                                     dictionary=dictionary, coherence='c_v')
    coherence.append((k, cm.get_coherence()))


# In[45]:


x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]


# In[46]:


plt.plot(x_val,y_val)
plt.scatter(x_val,y_val)
plt.title('Number of Topics vs. Coherence')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence')
plt.xticks(x_val)
plt.show()


# <div class="alert alert-info">
# <font color=black>
#     
# - The improvement stops to significantly improve after 15 topics.
# - It is not always where the highest Cv is, so we can try a couple to see which has the best result.
# - We'll try 15 and 23 here. 
# - Adding topics can help reveal further sub topics. Nonetheless, if the same words start to appear across multiple topics, the number of topics is too high.
# 
# </font>
# </div>

# In[47]:


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=15, id2word = dictionary, passes=40,\
               iterations=200,  chunksize = 10000, eval_every = None, random_state=0)


# In[48]:


Lda2 = gensim.models.ldamodel.LdaModel
ldamodel2 = Lda2(doc_term_matrix, num_topics=23, id2word = dictionary, passes=40,\
               iterations=200,  chunksize = 10000, eval_every = None, random_state=0)


# <div class="alert alert-info">
# <font color=black>
#     
# - `Passes`: The number of times model iterates through the whole corpus
# - `Iterations`: The number of iterations the model trains on each pass
# - `Chunk size`: Number of rows that are taken to train the model each 
# 
# </font>
# </div>

# In[49]:


# To show initial topics
ldamodel.show_topics(15, num_words=10, formatted=False)


# In[50]:


# To show initial topics
ldamodel2.show_topics(23, num_words=10, formatted=False)


# 23 topics yielded clearer results, so we'll go with this...

# # Relevancy
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-info">
# <font color=black>
#     
# - Sometimes, words that are ranked as top words for a given topic may be ranked high because they are globally frequent across text in a corpus. Relevancy score helps to prioritize terms that belong more exclusively to a given topic. This can increase interpretability even more. The relevance of term `w` to topic `k` is defined as:
#     
# $r(w,k| \lambda) = \lambda log(\phi_{kw}) +(1-\lambda)log(\frac{\phi_{kw}}{p_{kw}})$
# 
#     
# - where $\phi_{kw}$ is the probability of term w in topic k and $\frac{\phi_{kw}}{p_{kw}}$ is lift in term's probability within a topic to its marginal probability across the corpus (this helps discards globally frequent terms).
# - **How to tune it?**
#     - A lower lambda value gives more importance to the second term, which gives more importance to topic exclusivity. 
#     - A higher lambda value gives more importance to the first term, which gives less importance to topic exclusivity. 
# - We can use `Python’s pyLDAvis` interactive GUI to see this. Choose a value of 0.5 if undecided.
# 
# </font>
# </div>

# In[51]:


topic_data = pyLDAvis.gensim_models.prepare(
    ldamodel2, doc_term_matrix, dictionary, mds='pcoa')
pyLDAvis.display(topic_data)


# <div class="alert alert-info">
# <font color=black>
# 
# - The pyLDAvis tool also gives two other important pieces of information. 
# - The circles represent each topic.
# - The distance between the circles visualizes how related topics are to each other.
# - The above plot shows that our topics are quite distinct. 
# - The dimensionality reduction can be chosen as PCA or t-sne. The above example uses t-sne.
# - Additionally, the size of the circle represents how prevalent that topic is across the corpus of reviews. 
# 
# </font>
# </div>

# To extract the words for a given lambda:

# In[52]:


all_topics = {}
num_terms = 10  # Adjust number of words to represent each topic
lambd = 0.6  # Adjust this accordingly based on tuning above
for i in range(1, 24):  # Adjust this to reflect number of topics chosen for final LDA model
    topic = topic_data.topic_info[topic_data.topic_info.Category ==
                                  'Topic'+str(i)].copy()
    topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
    all_topics['Topic '+str(i)] = topic.sort_values(by='relevance',
                                                    ascending=False).Term[:num_terms].values


# In[53]:


pd.DataFrame(all_topics).T


# We can see here that most topics are quite clear. In this case, they are clustered into the different products being talked about in the reviews. Some clear topics are:
# - Printer products
# - Scanner proudcts
# - Printer Ink, pricing/quality
# - School stationary
# - Lamination
# - Packing tape
# - Mailing labels
# - Markers
# - Magnetic boards
# - Workstation
# - Binder
# - Paper products
# - Phone
# - Post-its
# - Storage/packing materials
# - Mouse/keyboard
# - Pencil
# - Cutting
# - File Organizer
# - Business card
# - Calculator
# - Mailing materials
# - Label maker

# # Clean-up folder
# <hr style="border:2px solid black"> </hr>

# In[ ]:


# We are not pushing to git big file
get_ipython().system('du -h reviews_Office_Products_5.json')


# In[ ]:


#!rm -rf reviews_Office_Products_5.json


# # References
# <hr style="border:2px solid black"> </hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - [Amazon office review dataset](https://jmcauley.ucsd.edu/data/amazon/)
# - [6 Tips for Interpretable Topic Models](https://towardsdatascience.com/6-tips-to-optimize-an-nlp-topic-model-for-interpretability-20742f3047e2)
# - [pyLDAvis paper](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)
# - [Latent Dirichlet Allocation and definition of hyperparameters](https://radimrehurek.com/gensim/models/ldamodel.html)
# - https://www.cs.cmu.edu/~epxing/Class/10708-15/slides/LDA_SC.pdf<br>
# - https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process
# 
# </font>
# </div>

# # Requirements
# <hr style="border:2px solid black"> </hr>

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -iv -m')


# In[ ]:




