#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#What-is-NER?" data-toc-modified-id="What-is-NER?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>What is NER?</a></span></li><li><span><a href="#Objective?" data-toc-modified-id="Objective?-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Objective?</a></span></li><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Gather-the-data" data-toc-modified-id="Gather-the-data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Gather the data</a></span></li><li><span><a href="#Extract-NER" data-toc-modified-id="Extract-NER-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Extract NER</a></span></li><li><span><a href="#Name-entity-linking" data-toc-modified-id="Name-entity-linking-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Name entity linking</a></span></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusions</a></span></li><li><span><a href="#References" data-toc-modified-id="References-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# <div class="alert alert-warning">
# <font color=black>
# 
# **What?** Name entity recognition = NER
# 
# </font>
# </div>

# # What is NER?

# <div class="alert alert-info">
# <font color=black>
# 
# - NER stands for Name Entity Recongition. 
# 
# </font>
# </div>

# # Objective?

# <div class="alert alert-info">
# <font color=black>
# 
# - The goal of this project is to learn and apply Named Entity Recognition to extract important entities(publicly traded companies in our example) and then link each entity with some information using a knowledge base(Nifty500 companies list).
# - We’ll get the textual data from RSS feeds on the internet, extract the names of buzzing stocks, and then pull their market price data to test the authenticity of the news before taking any position in those stocks.
# - **Essentially** To learn about what stocks are buzzing in the market and get their details on your dashboard is the goal for this project.
# 
# </font>
# </div>

# # Import modules

# In[32]:


import requests, spacy
from bs4 import BeautifulSoup
import warnings
import yfinance as yf
import pandas as pd
warnings.filterwarnings("ignore")


# # Gather the data

# <div class="alert alert-info">
# <font color=black>
# 
# - **Step #1** Get the entire XML document and we can use the requests library to do that.
# - **Step #2** Find out where in the XML file the data we are interest are. The headlines are present inside the `<title>` tag of the XML here.
# - **Step #3** Use `SpaCy` to extract the main entities from the headlines.
# 
# 
# - We'll be using this two news feeds:
#     - [Economic Times](https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms)
#     - [Money Control](https://www.moneycontrol.com/rss/buzzingstocks.xml)
# 
# </font>
# </div>

# In[3]:


XML_File_No1 = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
XML_File_No2 = "https://www.moneycontrol.com/rss/buzzingstocks.xml"
# Get the XML object
resp = requests.get(XML_File_No1)


# In[4]:


resp


# In[5]:


soup = BeautifulSoup(resp.content, features = "xml")


# In[10]:


headlines = soup.findAll("title")


# In[11]:


headlines


# # Extract NER

# <div class="alert alert-info">
# <font color=black>
# 
# - We’ll be using a **pre-trained** core language model from the spaCy library to extract the main entities in a headline.
# - spaCy has two major classes of pretrained language models that are trained on different sizes of textual data to give us state-of-the-art inferences.
#     - **Core Models** — for general-purpose basic NLP tasks.
#     - **Starter Models** — for niche applications that require transfer learning. Fine-tune our custom models without having to train the model from scratch. 
# 
# - Since our use case is basic in this tutorial, we are going to stick with the `en_core_web_sm` core model pipeline.
# 
# </font>
# </div>

# In[13]:


nlp = spacy.load('en_core_web_sm')


# In[15]:


# Let's see how it does with tokenization
for token in nlp(headlines[4].text):
    print(token)


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - A description of all POS can be found [here](https://spacy.io/models/en)
# - A description of the dependency graph can be found [here](https://spacy.io/models/en)
# 
# </font>
# </div>

# In[21]:


# Let's see how it does with tagging part of speech
for token in nlp(headlines[4].text):
    print(token, "POS? ", token.pos_, " DEPENDENCY GRAPH? ", token.dep_)


# In[24]:


# Visualize the relationship dependencies among the tokens
spacy.displacy.render(nlp(headlines[4].text), style='dep',jupyter=True, options={'distance': 120})


# In[25]:


# Important entities of the sentence, you can pass 'ent’ as style in the same code
spacy.displacy.render(nlp(headlines[4].text), style = 'ent',jupyter=True, options={'distance': 120})


# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - We have different tags for different entities like the day has DATE, Glasscoat has GPE which can be Countries/Cities/States. 
# - There are many entities we can extract, **which one are we interested in?**
# - We are majorly looking for entities that have `ORG` tag that’ll give us Companies, agencies, institutions, etc.
# 
# </font>
# </div>

# In[26]:


companies = []
for title in headlines:
    doc = nlp(title.text)
    for token in doc.ents:
        if token.label_ == 'ORG':
            companies.append(token.text)
        else:
            pass


# In[27]:


companies


# # Name entity linking

# <div class="alert alert-block alert-info">
# <font color=black>
# 
# - Of all the company we have, we'd like to select only some of them.
# 
# </font>
# </div>

# In[34]:


# Collect various market attributes of a stock into a dictionary
stock_dict = {
    'Org': [],
    'Symbol': [],
    'currentPrice': [],
    'dayHigh': [],
    'dayLow': [],
    'forwardPE': [],
    'dividendYield': []
}


# In[37]:


companies


# <div class="alert alert-info">
# <font color=black>
# 
# - We have the company names but in order to get their trading details, we’ll need the company’s trading stock symbol.
# - Since I am extracting the details and news of Indian Companies, I am going to use an external database of [Nifty 500 companies(a CSV file)](https://www1.nseindia.com/products/content/equities/indices/nifty_500.htm).
# - For every company, we’ll look it up in the list of companies using pandas, and then we’ll capture the stock market statistics using the yahoo `yfinance` library.
# 
# </font>
# </div>

# In[43]:


input_path = "../DATASETS/ind_nifty500list.csv"
stocks_df = pd.read_csv(input_path)
print('dimension: ', stocks_df.shape)
stocks_df.head()


# In[44]:


# For each company look it up and gather all market data on it
for company in companies:
    try:
        if stocks_df['Company Name'].str.contains(company).sum():
            symbol = stocks_df[stocks_df['Company Name'].\
                                str.contains(company)]['Symbol'].values[0]
            org_name = stocks_df[stocks_df['Company Name'].\
                                str.contains(company)]['Company Name'].values[0]
            stock_dict['Org'].append(org_name)
            stock_dict['Symbol'].append(symbol)
            # indian NSE stock symbols are stored with a .NS suffix in yfinance
            stock_info = yf.Ticker(symbol + ".NS").info
            stock_dict['currentPrice'].append(stock_info['currentPrice'])
            stock_dict['dayHigh'].append(stock_info['dayHigh'])
            stock_dict['dayLow'].append(stock_info['dayLow'])
            stock_dict['forwardPE'].append(stock_info['forwardPE'])
            stock_dict['dividendYield'].append(stock_info['dividendYield'])
        else:
            pass
    except:
        pass


# In[45]:


# Create a dataframe to display the buzzing stocks
pd.DataFrame(stock_dict)


# # Conclusions

# <div class="alert alert-danger">
# <font color=black>
# 
# - We have automatically extract name from the an xml file.
# - We have linked them to company we are interested in.
# - We have collect some important info about them.
# 
# </font>
# </div>

# # References

# <div class="alert alert-warning">
# <font color=black>
# 
# - https://www.kdnuggets.com/2021/09/-structured-financial-newsfeed-using-python-spacy-and-streamlit.html 
# 
# </font>
# </div>

# In[ ]:




