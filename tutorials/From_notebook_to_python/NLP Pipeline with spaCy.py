#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#NLP-Pipeline-with-spaCy" data-toc-modified-id="NLP-Pipeline-with-spaCy-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>NLP Pipeline with spaCy</a></span><ul class="toc-item"><li><span><a href="#Setup" data-toc-modified-id="Setup-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Setup</a></span><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#SpaCy-Language-Model-Installation" data-toc-modified-id="SpaCy-Language-Model-Installation-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>SpaCy Language Model Installation</a></span><ul class="toc-item"><li><span><a href="#English" data-toc-modified-id="English-2.1.2.1"><span class="toc-item-num">2.1.2.1&nbsp;&nbsp;</span>English</a></span></li><li><span><a href="#Spanish" data-toc-modified-id="Spanish-2.1.2.2"><span class="toc-item-num">2.1.2.2&nbsp;&nbsp;</span>Spanish</a></span></li><li><span><a href="#Validate-Installation" data-toc-modified-id="Validate-Installation-2.1.2.3"><span class="toc-item-num">2.1.2.3&nbsp;&nbsp;</span>Validate Installation</a></span></li></ul></li></ul></li><li><span><a href="#Get-Data" data-toc-modified-id="Get-Data-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Get Data</a></span></li><li><span><a href="#SpaCy-Pipeline-&amp;-Architecture" data-toc-modified-id="SpaCy-Pipeline-&amp;-Architecture-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>SpaCy Pipeline &amp; Architecture</a></span><ul class="toc-item"><li><span><a href="#The-Processing-Pipeline" data-toc-modified-id="The-Processing-Pipeline-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>The Processing Pipeline</a></span></li><li><span><a href="#Key-Data-Structures" data-toc-modified-id="Key-Data-Structures-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Key Data Structures</a></span></li></ul></li><li><span><a href="#SpaCy-in-Action" data-toc-modified-id="SpaCy-in-Action-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>SpaCy in Action</a></span><ul class="toc-item"><li><span><a href="#Create-&amp;-Explore-the-Language-Object" data-toc-modified-id="Create-&amp;-Explore-the-Language-Object-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Create &amp; Explore the Language Object</a></span></li><li><span><a href="#Explore-the-Pipeline" data-toc-modified-id="Explore-the-Pipeline-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Explore the Pipeline</a></span><ul class="toc-item"><li><span><a href="#Explore-Token-annotations" data-toc-modified-id="Explore-Token-annotations-2.4.2.1"><span class="toc-item-num">2.4.2.1&nbsp;&nbsp;</span>Explore <code>Token</code> annotations</a></span></li><li><span><a href="#Visualize-POS-Dependencies" data-toc-modified-id="Visualize-POS-Dependencies-2.4.2.2"><span class="toc-item-num">2.4.2.2&nbsp;&nbsp;</span>Visualize POS Dependencies</a></span></li><li><span><a href="#Visualize-Named-Entities" data-toc-modified-id="Visualize-Named-Entities-2.4.2.3"><span class="toc-item-num">2.4.2.3&nbsp;&nbsp;</span>Visualize Named Entities</a></span></li></ul></li><li><span><a href="#Read-BBC-Data" data-toc-modified-id="Read-BBC-Data-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Read BBC Data</a></span></li><li><span><a href="#Parse-first-article-through-Pipeline" data-toc-modified-id="Parse-first-article-through-Pipeline-2.4.4"><span class="toc-item-num">2.4.4&nbsp;&nbsp;</span>Parse first article through Pipeline</a></span></li><li><span><a href="#Detect-sentence-boundary" data-toc-modified-id="Detect-sentence-boundary-2.4.5"><span class="toc-item-num">2.4.5&nbsp;&nbsp;</span>Detect sentence boundary</a></span></li><li><span><a href="#Named-Entity-Recognition-with-textacy" data-toc-modified-id="Named-Entity-Recognition-with-textacy-2.4.6"><span class="toc-item-num">2.4.6&nbsp;&nbsp;</span>Named Entity-Recognition with textacy</a></span></li><li><span><a href="#N-Grams-with-textacy" data-toc-modified-id="N-Grams-with-textacy-2.4.7"><span class="toc-item-num">2.4.7&nbsp;&nbsp;</span>N-Grams with textacy</a></span></li><li><span><a href="#The-spaCy-streaming-Pipeline-API" data-toc-modified-id="The-spaCy-streaming-Pipeline-API-2.4.8"><span class="toc-item-num">2.4.8&nbsp;&nbsp;</span>The spaCy streaming Pipeline API</a></span></li><li><span><a href="#Multi-language-Features" data-toc-modified-id="Multi-language-Features-2.4.9"><span class="toc-item-num">2.4.9&nbsp;&nbsp;</span>Multi-language Features</a></span><ul class="toc-item"><li><span><a href="#Create-a-Spanish-Language-Object" data-toc-modified-id="Create-a-Spanish-Language-Object-2.4.9.1"><span class="toc-item-num">2.4.9.1&nbsp;&nbsp;</span>Create a Spanish Language Object</a></span></li><li><span><a href="#Read-bilingual-TED2013-samples" data-toc-modified-id="Read-bilingual-TED2013-samples-2.4.9.2"><span class="toc-item-num">2.4.9.2&nbsp;&nbsp;</span>Read bilingual TED2013 samples</a></span></li><li><span><a href="#Sentence-Boundaries-English-vs-Spanish" data-toc-modified-id="Sentence-Boundaries-English-vs-Spanish-2.4.9.3"><span class="toc-item-num">2.4.9.3&nbsp;&nbsp;</span>Sentence Boundaries English vs Spanish</a></span></li><li><span><a href="#POS-Tagging-English-vs-Spanish" data-toc-modified-id="POS-Tagging-English-vs-Spanish-2.4.9.4"><span class="toc-item-num">2.4.9.4&nbsp;&nbsp;</span>POS Tagging English vs Spanish</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Clean-up-folder" data-toc-modified-id="Clean-up-folder-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Clean-up folder</a></span></li><li><span><a href="#References" data-toc-modified-id="References-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** NLP Pipeline with spaCy
# 
# </font>
# </div>

# # NLP Pipeline with spaCy

# [spaCy](https://spacy.io/) is a widely used python library with a comprehensive feature set for fast text processing in multiple languages. 
# 

# ## Setup

# ### Imports

# In[1]:


import sys
from pathlib import Path

import pandas as pd

import spacy
from spacy import displacy
from textacy.extract import ngrams, entities


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### SpaCy Language Model Installation
# 
# In addition to the `spaCy` library, we need [language models](https://spacy.io/usage/models).

# #### English

# Only need to run once.

# In[3]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy download en_core_web_sm\n\n# more comprehensive models:\n# {sys.executable} -m spacy download en_core_web_md\n# {sys.executable} -m spacy download en_core_web_lg\n')


# #### Spanish

# [Spanish language models](https://spacy.io/models/es#es_core_news_sm) trained on [AnCora Corpus](http://clic.ub.edu/corpus/) and [WikiNER](http://schwa.org/projects/resources/wiki/Wikiner)

# Only need to run once.

# In[4]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy download es_core_news_sm\n\n# more comprehensive model:\n# {sys.executable} -m spacy download es_core_news_md\n')


# Create shortcut names

# In[5]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy link en_core_web_sm en;\npython -m spacy link es_core_news_sm es;\n\n')


# #### Validate Installation

# In[6]:


# validate installation
get_ipython().system('{sys.executable} -m spacy validate')


# ## Get Data

# In[7]:


DATA_DIR = Path('./')


# - [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files ([download](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip))
#     - Data already included in [data](../data) directory, just unzip before first-time use.
# - [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus of TED talk subtitles in 15 langugages (sample provided) in `results/TED` subfolder of this directory.

# ## SpaCy Pipeline & Architecture

# ### The Processing Pipeline
# 
# When you call a spaCy model on a text, spaCy 
# 
# 1) tokenizes the text to produce a `Doc` object. 
# 
# 2) passes the `Doc` object through the processing pipeline that may be customized, and for the default models consists of
# - a tagger, 
# - a parser and 
# - an entity recognizer. 
# 
# Each pipeline component returns the processed Doc, which is then passed on to the next component.

# ![image.png](attachment:image.png)

# ### Key Data Structures
# 
# The central data structures in spaCy are the **Doc** and the **Vocab**. Text annotations are also designed to allow a single source of truth:
# 
# - The **`Doc`** object owns the sequence of tokens and all their annotations. `Span` and `Token` are views that point into it. It is constructed by the `Tokenizer`, and then modified in place by the components of the pipeline. 
# - The **`Vocab`** object owns a set of look-up tables that make common information available across documents. 
# - The **`Language`** object coordinates these components. It takes raw text and sends it through the pipeline, returning an annotated document. It also orchestrates training and serialization.

# ![image.png](attachment:image.png)

# ## SpaCy in Action

# ### Create & Explore the Language Object

# Once installed and linked, we can instantiate a spaCy language model and then call it on a document. As a result, spaCy produces a Doc object that tokenizes the text and processes it according to configurable pipeline components that by default consist of a tagger, a parser, and a named-entity recognizer.

# In[8]:


"""nlp = spacy.load('en') """
nlp = spacy.load("en_core_web_sm")


# In[9]:


type(nlp)


# In[10]:


nlp.lang


# In[11]:


spacy.info('en_core_web_sm')


# In[12]:


def get_attributes(f):
    print([a for a in dir(f) if not a.startswith('_')], end=' ')


# In[13]:


get_attributes(nlp)


# ### Explore the Pipeline

# Let’s illustrate the pipeline using a simple sentence:

# In[14]:


sample_text = 'Apple is looking at buying U.K. startup for $1 billion'
doc = nlp(sample_text)


# In[15]:


get_attributes(doc)


# In[16]:


doc.is_parsed


# In[17]:


doc.is_sentenced


# In[18]:


doc.is_tagged


# In[19]:


doc.text


# In[20]:


get_attributes(doc.vocab)


# In[21]:


doc.vocab.length


# #### Explore `Token` annotations

# The parsed document content is iterable and each element has numerous attributes produced by the processing pipeline. The below sample illustrates how to access the following attributes:

# In[22]:


pd.Series([token.text for token in doc])


# In[23]:


pd.DataFrame([[t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.shape_, t.is_alpha, t.is_stop]
              for t in doc],
             columns=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop'])


# #### Visualize POS Dependencies

# We can visualize the syntactic dependency in a browser or notebook

# In[24]:


options = {'compact': True, 'bg': 'white',
           'color': 'black', 'font': 'Source Sans Pro', 'notebook': True}


# In[25]:


displacy.render(doc, style='dep', options=options)


# #### Visualize Named Entities

# In[26]:


displacy.render(doc, style='ent', jupyter=True)


# ### Read BBC Data

# We will now read a larger set of 2,225 BBC News articles (see GitHub for data source details) that belong to five categories and are stored in individual text files. We 
# - call the .glob() method of the pathlib’s Path object, 
# - iterate over the resulting list of paths, 
# - read all lines of the news article excluding the heading in the first line, and 
# - append the cleaned result to a list

# - Download data from this [link](https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/data/bbc.zip)

# In[27]:


files = (DATA_DIR / 'bbc').glob('**/*.txt')
bbc_articles = []
for i, file in enumerate(sorted(list(files))):
    with file.open(encoding='latin1') as f:
        lines = f.readlines()
        body = ' '.join([l.strip() for l in lines[1:]]).strip()
        bbc_articles.append(body)


# In[28]:


len(bbc_articles)


# In[29]:


bbc_articles[0]


# ### Parse first article through Pipeline

# In[30]:


nlp.pipe_names


# In[31]:


doc = nlp(bbc_articles[0])
type(doc)


# ### Detect sentence boundary
# Sentence boundaries are calculated from the syntactic parse tree, so features such as punctuation and capitalisation play an important but non-decisive role in determining the sentence boundaries. 
# 
# Usually this means that the sentence boundaries will at least coincide with clause boundaries, even given poorly punctuated text.

# spaCy computes sentence boundaries from the syntactic parse tree so that punctuation and capitalization play an important but not decisive role. As a result, boundaries will coincide with clause boundaries, even for poorly punctuated text.
# 
# We can access the parsed sentences using the .sents attribute:

# In[32]:


sentences = [s for s in doc.sents]
sentences[:3]


# In[33]:


get_attributes(sentences[0])


# In[34]:


pd.DataFrame([[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[0]], 
             columns=['Token', 'POS Tag', 'Meaning']).head(15)


# In[35]:


options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Source Sans Pro'}
displacy.render(sentences[0].as_doc(), style='dep', jupyter=True, options=options)


# In[36]:


for t in sentences[0]:
    if t.ent_type_:
        print('{} | {} | {}'.format(t.text, t.ent_type_, spacy.explain(t.ent_type_)))


# In[37]:


displacy.render(sentences[0].as_doc(), style='ent', jupyter=True)


# ### Named Entity-Recognition with textacy

# spaCy enables named entity recognition using the .ent_type_ attribute:

# Textacy makes access to the named entities that appear in the first article easy:

# In[38]:


entities = [e.text for e in entities(doc)]
pd.Series(entities).value_counts().head()


# ### N-Grams with textacy

# N-grams combine N consecutive tokens. This can be useful for the bag-of-words model because, depending on the textual context, treating, e.g, ‘data scientist’ as a single token may be more meaningful than the two distinct tokens ‘data’ and ‘scientist’.
# 
# Textacy makes it easy to view the ngrams of a given length n occurring with at least min_freq times:

# In[39]:


pd.Series([n.text for n in ngrams(doc, n=2, min_freq=2)]).value_counts()


# ### The spaCy streaming Pipeline API

# To pass a larger number of documents through the processing pipeline, we can use spaCy’s streaming API as follows:

# In[43]:


iter_texts = (bbc_articles[i] for i in range(len(bbc_articles)))
for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_process=-1)):
    if i % 100 == 0:
        print(i, end = ' ')
    assert doc.is_parsed


# ### Multi-language Features

# spaCy includes trained language models for English, German, Spanish, Portuguese, French, Italian and Dutch, as well as a multi-language model for named-entity recognition. Cross-language usage is straightforward since the API does not change.
# 
# We will illustrate the Spanish language model using a parallel corpus of TED talk subtitles. For this purpose, we instantiate both language models

# #### Create a Spanish Language Object

# In[44]:


model = {}
for language in ['en', 'es']:
    model[language] = spacy.load(language) 


# #### Read bilingual TED2013 samples

# In[ ]:


text = {}
path = Path('data', 'TED')
for language in ['en', 'es']:
    file_name = path /  'TED2013_sample.{}'.format(language)
    text[language] = file_name.read_text()


# #### Sentence Boundaries English vs Spanish

# In[ ]:


parsed, sentences = {}, {}
for language in ['en', 'es']:
    parsed[language] = model[language](text[language])
    sentences[language] = list(parsed[language].sents)
    print('Sentences:', language, len(sentences[language]))


# In[ ]:


for i, (en, es) in enumerate(zip(sentences['en'], sentences['es']), 1):
    print('\n', i)
    print('English:\t', en)
    print('Spanish:\t', es)
    if i > 5: 
        break


# #### POS Tagging English vs Spanish

# In[ ]:


pos = {}
for language in ['en', 'es']:
    pos[language] = pd.DataFrame([[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[language][0]],
                                 columns=['Token', 'POS Tag', 'Meaning'])


# In[ ]:


bilingual_parsed = pd.concat([pos['en'], pos['es']], axis=1)
bilingual_parsed.head(15)


# In[ ]:


displacy.render(sentences['es'][0].as_doc(), style='dep', jupyter=True, options=options)


# # Clean-up folder
# <hr style = "border:2px solid black" ></hr>

# In[48]:


import shutil
get_ipython().system('shutil.rmtree("./bbc")')


# # References
# <hr style = "border:2px solid black" ></hr>

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/14_working_with_text_data/01_nlp_pipeline_with_spaCy.ipynb
# - Jansen, Stefan. Hands-On Machine Learning for Algorithmic Trading: Design and implement investment strategies based on smart algorithms that learn from data using Python. Packt Publishing Ltd, 2018.
# 
# </font>
# </div>

# In[ ]:




