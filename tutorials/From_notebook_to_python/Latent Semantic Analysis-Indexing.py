#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Latent-Semantic-Analysis/Indexing" data-toc-modified-id="Latent-Semantic-Analysis/Indexing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Latent Semantic Analysis/Indexing</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-BBC-data" data-toc-modified-id="Load-BBC-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load BBC data</a></span></li><li><span><a href="#Convert-to-DataFrame" data-toc-modified-id="Convert-to-DataFrame-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Convert to DataFrame</a></span><ul class="toc-item"><li><span><a href="#Create-Train-&amp;-Test-Sets" data-toc-modified-id="Create-Train-&amp;-Test-Sets-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Create Train &amp; Test Sets</a></span><ul class="toc-item"><li><span><a href="#Vectorize-train-&amp;-test-sets" data-toc-modified-id="Vectorize-train-&amp;-test-sets-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Vectorize train &amp; test sets</a></span></li><li><span><a href="#Get-token-count" data-toc-modified-id="Get-token-count-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Get token count</a></span></li></ul></li></ul></li><li><span><a href="#Latent-Semantic-Analysis" data-toc-modified-id="Latent-Semantic-Analysis-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Latent Semantic Analysis</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Explore-Topics" data-toc-modified-id="Explore-Topics-6.0.1"><span class="toc-item-num">6.0.1&nbsp;&nbsp;</span>Explore Topics</a></span><ul class="toc-item"><li><span><a href="#Topic-Weights-for-sample-article" data-toc-modified-id="Topic-Weights-for-sample-article-6.0.1.1"><span class="toc-item-num">6.0.1.1&nbsp;&nbsp;</span>Topic Weights for sample article</a></span></li><li><span><a href="#Average-topic-weight-per-category" data-toc-modified-id="Average-topic-weight-per-category-6.0.1.2"><span class="toc-item-num">6.0.1.2&nbsp;&nbsp;</span>Average topic weight per category</a></span></li><li><span><a href="#Topics-weights-of-most-frequent-words" data-toc-modified-id="Topics-weights-of-most-frequent-words-6.0.1.3"><span class="toc-item-num">6.0.1.3&nbsp;&nbsp;</span>Topics weights of most frequent words</a></span></li><li><span><a href="#Most-important-words-by-topic" data-toc-modified-id="Most-important-words-by-topic-6.0.1.4"><span class="toc-item-num">6.0.1.4&nbsp;&nbsp;</span>Most important words by topic</a></span></li><li><span><a href="#Topics-weights-for-test-set" data-toc-modified-id="Topics-weights-for-test-set-6.0.1.5"><span class="toc-item-num">6.0.1.5&nbsp;&nbsp;</span>Topics weights for test set</a></span></li></ul></li><li><span><a href="#Categories-in-2D" data-toc-modified-id="Categories-in-2D-6.0.2"><span class="toc-item-num">6.0.2&nbsp;&nbsp;</span>Categories in 2D</a></span></li></ul></li></ul></li><li><span><a href="#Strenghts-&amp;-Weaknesses" data-toc-modified-id="Strenghts-&amp;-Weaknesses-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Strenghts &amp; Weaknesses</a></span></li><li><span><a href="#Clean-up-folder" data-toc-modified-id="Clean-up-folder-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Clean-up folder</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Latent Semantic Analysis-Indexing
# 
# </font>
# </div>

# # Latent Semantic Analysis/Indexing
# <hr style="border:2px solid black"> </hr>

# Latent Semantic Analysis set out to improve the results of queries that omitted relevant documents containing synonyms of query terms. Its aimed to model the relationships between documents and terms to be able to predict that a term should be associated with a document, even though, because of variability in word use, no such association was observed.
# 
# LSI uses linear algebra to find a given number k of latent topics by decomposing the DTM. More specifically, it uses the Singular Value Decomposition (SVD) to find the best lower-rank DTM approximation using k singular values & vectors. In other words, LSI is an application of the unsupervised learning techniques of dimensionality reduction we encountered in chapter 12 (with some additional detail). The authors experimented with hierarchical clustering but found it too restrictive to explicitly model the document-topic and topic-term relationships or capture associations of documents or terms with several topics.

# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
from random import randint
import numpy as np
import pandas as pd

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')


# # Load BBC data
# <hr style="border:2px solid black"> </hr>

# We will illustrate the application of LSI using the BBC articles data that we introduced in the last chapter (13) because they are both small to permit quick training and allow us to compare topic assignments to category labels.

# - Download data from this [link](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/data/bbc.zip)

# In[4]:


# change to your data path if necessary
DATA_DIR = Path('../data')


# In[5]:


path = DATA_DIR / 'bbc'
files = sorted(list(path.glob('**/*.txt')))
doc_list = []
for i, file in enumerate(files):
    with open(str(file), encoding='latin1') as f:
        topic = file.parts[-2]
        lines = f.readlines()
        heading = lines[0].strip()
        body = ' '.join([l.strip() for l in lines[1:]])
        doc_list.append([topic.capitalize(), heading, body])


# # Convert to DataFrame
# <hr style="border:2px solid black"> </hr>

# In[6]:


docs = pd.DataFrame(doc_list, columns=['Category', 'Heading', 'Article'])
docs.info()


# ## Create Train & Test Sets

# We begin by loading the documents and creating a train and (stratified) test set with 50 articles. Then, we vectorize the data using the TfidfVectorizer to obtain weighted DTM counts and filter out words that appear in less than 1% or more than 25% of the documents as well as generic stopwords to obtain a vocabulary of around 2,900 words:

# In[7]:


train_docs, test_docs = train_test_split(docs,
                                         stratify=docs.Category,
                                         test_size=50,
                                         random_state=42)


# In[8]:


train_docs.shape, test_docs.shape


# In[9]:


pd.Series(test_docs.Category).value_counts()


# ### Vectorize train & test sets

# In[10]:


vectorizer = TfidfVectorizer(max_df=.25,
                             min_df=.01,
                             stop_words='english',
                             binary=False)
train_dtm = vectorizer.fit_transform(train_docs.Article)
train_dtm


# In[11]:


test_dtm = vectorizer.transform(test_docs.Article)
test_dtm


# ### Get token count

# In[12]:


train_token_count = train_dtm.sum(0).A.squeeze()
tokens = vectorizer.get_feature_names()
word_count = pd.Series(train_token_count,
                       index=tokens).sort_values(ascending=False)
word_count.head(10)


# # Latent Semantic Analysis
# <hr style="border:2px solid black"> </hr>

# We use sklearn’s TruncatedSVD class that only computes the k largest singular values to reduce the dimensionality of the document-term matrix. The deterministic arpack algorithm delivers an exact solution but the default ‘randomized’ implementation is more efficient for large matrices. 
# 
# We compute five topics to match the five categories, which explain only 5.4% of the total DTM variance so higher values would be reasonable.

# In[13]:


n_components = 5
topic_labels = ['Topic {}'.format(i) for i in range(1, n_components + 1)]


# In[14]:


svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
svd.fit(train_dtm)


# In[15]:


svd.singular_values_


# In[16]:


svd.explained_variance_ratio_.sum()


# ### Explore Topics 

# LSI identifies a new orthogonal basis for the document-term matrix that reduces the rank to the number of desired topics. 
# 
# The `.transform()` method of the trained svd object projects the documents into the new topic space that is the result of reducing the dimensionality of the document vectors and corresponds to the UTΣT transformation explained in the book.

# In[17]:


train_doc_topics = svd.transform(train_dtm)


# In[18]:


train_doc_topics.shape


# #### Topic Weights for sample article

# We can sample an article to view its location in the topic space. We draw a ‘Politics’ article that is most (positively) associated with topics 1 and 2 (and 3):

# In[19]:


i = randint(0, len(train_docs))
(train_docs.iloc[i, :2].append(
    pd.Series(train_doc_topics[i], index=topic_labels)))


# #### Average topic weight per category

# In[20]:


train_result = pd.DataFrame(data=train_doc_topics,
                            columns=topic_labels,
                            index=train_docs.Category)
train_result.groupby(level='Category').mean().plot.bar(figsize=(14, 5), rot=0);


# #### Topics weights of most frequent words

# In[21]:


topics = pd.DataFrame(svd.components_.T, index=tokens, columns=topic_labels)
topics.loc[word_count.head(10).index]


# #### Most important words by topic

# We can also display the words that are most closely associated with each topic (in absolute terms). The topics appear to capture some semantic information but are not clearly differentiated

# In[22]:


fig, ax = plt.subplots(figsize=(12, 4))
top_words, top_vals = pd.DataFrame(), pd.DataFrame()
for topic, words_ in topics.items():
    top10 = words_.abs().nlargest(10).index
    vals = words_.loc[top10].values
    top_vals[topic] = vals
    top_words[topic] = top10.tolist()
sns.heatmap(pd.DataFrame(top_vals),
            annot=top_words,
            fmt='',
            center=0,
            cmap=sns.diverging_palette(0, 255, sep=1, n=256),
            ax=ax)
ax.set_title('Top Words per Topic')
sns.despine()
fig.tight_layout()


# #### Topics weights for test set

# The topic assignments for this sample align with the average topic weights for each category illustrated below (Politics is the leftmost). They illustrate how LSI expresses the k topics as directions in a k-dimensional space (below you find a projection of the average topic assignments per category into two-dimensional space). 
# 
# Each category is clearly defined, and the test assignments match with train assignments. However, the weights are both positive and negative, making it more difficult to interpret the topics.

# In[23]:


test_eval = pd.DataFrame(data=svd.transform(test_dtm),
                         columns=topic_labels,
                         index=test_docs.Category)


# In[24]:


sns.set(font_scale=1.3)
result = pd.melt(train_result.assign(Data='Train').append(
    test_eval.assign(Data='Test')).reset_index(),
                 id_vars=['Data', 'Category'],
                 var_name='Topic',
                 value_name='Weight')

g = sns.catplot(x='Category',
                y='Weight',
                hue='Topic',
                row='Data',
                kind='bar',
                data=result,
                aspect=3.5);


# ### Categories in 2D

# The below plot shows the projections of the five topics into a 2D space.

# In[25]:


pca = PCA(n_components=2)
svd2d = pd.DataFrame(pca.fit_transform(train_result),
                     columns=['PC1',
                              'PC2']).assign(Category=train_docs.Category)
categories_2d = svd2d.groupby('Category').mean()


# In[26]:


plt.quiver(np.zeros(5),
           np.zeros(5),
           categories_2d.PC1.values,
           categories_2d.PC2.values,
           scale=.035)
plt.title('Topic Directions in 2D Space');


# # Strenghts & Weaknesses
# <hr style="border:2px solid black"> </hr>

# The benefits of LSI include the removal of noise and mitigation of the curse of dimensionality, while also capturing some semantics and performing a clustering of both documents and terms.
# 
# However, the results of LSI are difficult to interpret because topics are word vectors with both positive and negative entries. In addition, there is no underlying model that would permit the evaluation of fit and provide guidance when selecting the number of dimensions or topics.

# # Clean-up folder
# <hr style = "border:2px solid black" ></hr>

# In[ ]:


import shutil
get_ipython().system('shutil.rmtree("./bbc")')


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/15_topic_modeling/01_latent_semantic_indexing.ipynb
# - Jansen, Stefan. Hands-On Machine Learning for Algorithmic Trading: Design and implement investment strategies based on smart algorithms that learn from data using Python. Packt Publishing Ltd, 2018.
# 
# 
# </font>
# </div>

# In[ ]:




