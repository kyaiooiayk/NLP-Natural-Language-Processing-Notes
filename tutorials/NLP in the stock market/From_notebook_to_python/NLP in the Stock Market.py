#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Modules-installtion-issues" data-toc-modified-id="Modules-installtion-issues-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Modules installtion issues</a></span></li><li><span><a href="#Import" data-toc-modified-id="Import-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Import</a></span></li><li><span><a href="#Download-NLP-Corpora" data-toc-modified-id="Download-NLP-Corpora-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Download NLP Corpora</a></span></li><li><span><a href="#Get-10ks" data-toc-modified-id="Get-10ks-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Get 10ks</a></span></li><li><span><a href="#Get-list-of-10-ks" data-toc-modified-id="Get-list-of-10-ks-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Get list of 10-ks</a></span></li><li><span><a href="#Download-10-ks" data-toc-modified-id="Download-10-ks-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Download 10-ks</a></span></li><li><span><a href="#Get-documens" data-toc-modified-id="Get-documens-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Get documens</a></span></li><li><span><a href="#Get-Document-Types" data-toc-modified-id="Get-Document-Types-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Get Document Types</a></span></li><li><span><a href="#Document-clean-up" data-toc-modified-id="Document-clean-up-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Document clean-up</a></span></li><li><span><a href="#Lemmatize" data-toc-modified-id="Lemmatize-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Lemmatize</a></span></li><li><span><a href="#Remove-Stopwords" data-toc-modified-id="Remove-Stopwords-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Remove Stopwords</a></span></li><li><span><a href="#Loughran-McDonald-Sentiment-Word-Lists" data-toc-modified-id="Loughran-McDonald-Sentiment-Word-Lists-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Loughran McDonald Sentiment Word Lists</a></span></li><li><span><a href="#Bag-of-Words" data-toc-modified-id="Bag-of-Words-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Bag of Words</a></span></li><li><span><a href="#Jaccard-Similarity" data-toc-modified-id="Jaccard-Similarity-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Jaccard Similarity</a></span></li><li><span><a href="#TFIDF" data-toc-modified-id="TFIDF-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>TFIDF</a></span></li><li><span><a href="#Cosine-Similarity" data-toc-modified-id="Cosine-Similarity-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>Cosine Similarity</a></span></li><li><span><a href="#Price-Data" data-toc-modified-id="Price-Data-18"><span class="toc-item-num">18&nbsp;&nbsp;</span>Price Data</a></span></li><li><span><a href="#Dict-to-DataFrame" data-toc-modified-id="Dict-to-DataFrame-19"><span class="toc-item-num">19&nbsp;&nbsp;</span>Dict to DataFrame</a></span></li><li><span><a href="#Alphalens-Format" data-toc-modified-id="Alphalens-Format-20"><span class="toc-item-num">20&nbsp;&nbsp;</span>Alphalens Format</a></span></li><li><span><a href="#References" data-toc-modified-id="References-21"><span class="toc-item-num">21&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # Introduction

# In[1]:


"""
What? NLP in the Stock Market
"""


# # Modules installtion issues

# In[2]:


"""
Attention if you are on MAC please make sure you do:
pip install bs4
pip install html5lib
pip install lxml

Reference: https://stackoverflow.com/questions/24398302/bs4-featurenotfound-couldnt-find-a-tree-builder-with-the-features-you-requeste
"""


# # Import

# In[46]:


import nltk, os
import numpy as np
import pandas as pd
import pickle
import pprint
import project_helper
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Download NLP Corpora

# In[4]:


"""
You'll need two corpora to run this project: 
    [1] stopwords corpus for removing stopwords 
    [2] wordnet for lemmatizing.
"""


# In[ ]:


nltk.download('stopwords')
nltk.download('wordnet')


# # Get 10ks

# In[6]:


"""
Financial documents such as 10-k forms can be used to forecast stock movements. 10-k forms are annual reports 
filed by companies to provide a comprehensive summary of their financial performance.

10-k documents include information such as company history, organizational structure, executive compensation, 
equity, subsidiaries, and audited financial statements. Each company has an unique CIK (Central Index Key).
In this case  we selected 7 of them to keep the computation low.
"""


# In[7]:


cik_lookup = {
    'AMZN': '0001018724',
    'BMY': '0000014272',   
    'CNP': '0001130310',
    'CVX': '0000093410',
    'FL': '0000850209',
    'FRT': '0000034903',
    'HON': '0000773840'}

additional_cik = {
    'AEP': '0000004904',
    'AXP': '0000004962',
    'BA': '0000012927', 
    'BK': '0001390777',
    'CAT': '0000018230',
    'DE': '0000315189',
    'DIS': '0001001039',
    'DTE': '0000936340',
    'ED': '0001047862',
    'EMR': '0000032604',
    'ETN': '0001551182',
    'GE': '0000040545',
    'IBM': '0000051143',
    'IP': '0000051434',
    'JNJ': '0000200406',
    'KO': '0000021344',
    'LLY': '0000059478',
    'MCD': '0000063908',
    'MO': '0000764180',
    'MRK': '0000310158',
    'MRO': '0000101778',
    'PCG': '0001004980',
    'PEP': '0000077476',
    'PFE': '0000078003',
    'PG': '0000080424',
    'PNR': '0000077360',
    'SYY': '0000096021',
    'TXN': '0000097476',
    'UTX': '0000101829',
    'WFC': '0000072971',
    'WMT': '0000104169',
    'WY': '0000106535',
    'XOM': '0000034088'}


# # Get list of 10-ks

# In[8]:


"""
The SEC has a limit on the number of calls you can make to the website per second. In order to avoid hiding that 
limit, we've created the SecAPI class. This will cache data from the SEC and prevent you from going over the limit.
"""


# In[9]:


sec_api = project_helper.SecAPI()


# In[10]:


def get_sec_data(cik, doc_type, start=0, count=60):
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)]

    return entries


# In[11]:


"""
Let's pull the list using the get_sec_data function, then display some of the results. For displaying some of 
the data, we'll use Amazon as an example.
"""


# In[12]:


example_ticker = 'AMZN'
sec_data = {}

for ticker, cik in cik_lookup.items():
    sec_data[ticker] = get_sec_data(cik, '10-K')
    
pprint.pprint(sec_data[example_ticker][:5])


# # Download 10-ks

# In[13]:


"""
As you see, this is a list of urls. These urls point to a file that contains metadata related to each filling. 
Since we don't care about the metadata, we'll pull the filling by replacing the url with the filling url.
"""


# In[14]:


raw_fillings_by_ticker = {}

for ticker, data in sec_data.items():
    raw_fillings_by_ticker[ticker] = {}
    for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
        if (file_type == '10-K'):
            file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
            
            raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)


print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[example_ticker].values()))[:1000]))


# # Get documens

# In[15]:


"""
With theses fillings downloaded, we want to break them into their associated documents. These documents are 
sectioned off in the fillings with the tags <DOCUMENT> for the start of each document and </DOCUMENT> for the 
end of each document. There's no overlap with these documents, so each </DOCUMENT> tag should come after the 
<DOCUMENT> with no <DOCUMENT> tag in between.

Implement get_documents to return a list of these documents from a filling. Make sure not to include the tag in 
the returned document text.
"""


# In[16]:


def get_documents(text):
    """
    Extract the documents from the text

    Parameters
    ----------
    text : str
        The text with the document strings inside

    Returns
    -------
    extracted_docs : list of str
        The document strings found in `text`
    """

    # TODO: Implement
    extracted_docs = []

    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(text)]

    for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is):
        extracted_docs.append(text[doc_start_i:doc_end_i])

    return extracted_docs


# In[17]:


"""
With the get_documents function implemented, let's extract all the documents.
"""


# In[18]:


filling_documents_by_ticker = {}

for ticker, raw_fillings in raw_fillings_by_ticker.items():
    filling_documents_by_ticker[ticker] = {}
    for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
        filling_documents_by_ticker[ticker][file_date] = get_documents(filling)


print('\n\n'.join([
    'Document {} Filed on {}:\n{}...'.format(doc_i, file_date, doc[:200])
    for file_date, docs in filling_documents_by_ticker[example_ticker].items()
    for doc_i, doc in enumerate(docs)][:3]))


# # Get Document Types

# In[19]:


"""
Now that we have all the documents, we want to find the 10-k form in this 10-k filing. Implement the 
get_document_type function to return the type of document given. The document type is located on a line 
with the <TYPE> tag. For example, a form of type "TEST" would have the line <TYPE>TEST. Make sure to return 
the type as lowercase, so this example would be returned as "test".
"""


# In[20]:


def get_document_type(doc):
    """
    Return the document type lowercased

    Parameters
    ----------
    doc : str
        The document string

    Returns
    -------
    doc_type : str
        The document type lowercased
    """

    # TODO: Implement
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_type = type_pattern.findall(doc)[0][len('<TYPE>'):]

    return doc_type.lower()


# In[21]:


"""
With the get_document_type function, we'll filter out all non 10-k documents.
"""


# In[22]:


ten_ks_by_ticker = {}

for ticker, filling_documents in filling_documents_by_ticker.items():
    ten_ks_by_ticker[ticker] = []
    for file_date, documents in filling_documents.items():
        for document in documents:
            if get_document_type(document) == '10-k':
                ten_ks_by_ticker[ticker].append({
                    'cik': cik_lookup[ticker],
                    'file': document,
                    'file_date': file_date})


project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], [
                                'cik', 'file', 'file_date'])


# # Document clean-up

# In[23]:


"""
As you can see, the text for the documents are very messy. To clean this up, we'll remove the html and lowercase 
all the text.
"""


# In[24]:


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    
    return text


# In[25]:


for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_clean'] = clean_text(ten_k['file'])


project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_clean'])


# # Lemmatize

# In[26]:


"""
With the text cleaned up, it's time to distill the verbs down. Implement the lemmatize_words function to lemmatize 
verbs in the list of words provided.
"""


# In[27]:


def lemmatize_words(words):
    """
    Lemmatize words 

    Parameters
    ----------
    words : list of str
        List of words

    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """

    # TODO: Implement
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v')
                        for word in words]

    return lemmatized_words


# In[28]:


word_pattern = re.compile('\w+')

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = lemmatize_words(word_pattern.findall(ten_k['file_clean']))


project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_lemma'])


# # Remove Stopwords

# In[29]:


lemma_english_stopwords = lemmatize_words(stopwords.words('english'))

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]


print('Stop Words Removed')


# # Loughran McDonald Sentiment Word Lists

# In[ ]:


"""
We'll be using the Loughran and McDonald sentiment word lists. These word lists cover the following sentiment:

    Negative
    Positive
    Uncertainty
    Litigious
    Constraining
    Superfluous
    Modal

This will allow us to do the sentiment analysis on the 10-ks. Let's first load these word lists. We'll be 
looking into a few of these sentiments.

The dataset was retrievd from here: https://sraf.nd.edu/textual-analysis/resources/#LM%20Sentiment%20Word%20Lists
"""


# In[32]:


sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']

sentiment_df = pd.read_csv(os.path.join('./LoughranMcDonald_MasterDictionary_2018.csv'))
sentiment_df.columns = [column.lower() for column in sentiment_df.columns] # Lowercase the columns for ease of use

# Remove unused information
sentiment_df = sentiment_df[sentiments + ['word']]
sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

# Apply the same preprocessing to these words as the 10-k words
sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
sentiment_df = sentiment_df.drop_duplicates('word')


sentiment_df.head()


# # Bag of Words

# In[ ]:


"""
Using the sentiment word lists, let's generate sentiment bag of words from the 10-k documents. Implement 
get_bag_of_words to generate a bag of words that counts the number of sentiment words in each doc. You can 
ignore words that are not in sentiment_words.
"""


# In[34]:


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag of words from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    
    # TODO: Implement
    vec = CountVectorizer(vocabulary=sentiment_words)
    vectors = vec.fit_transform(docs)
    words_list = vec.get_feature_names()
    bag_of_words = np.zeros([len(docs), len(words_list)])
    
    for i in range(len(docs)):
        bag_of_words[i] = vectors[i].toarray()[0]

    return bag_of_words.astype(int)


# In[ ]:


"""
Using the get_bag_of_words function, we'll generate a bag of words for all the documents.
"""


# In[36]:


sentiment_bow_ten_ks = {}

for ticker, ten_ks in ten_ks_by_ticker.items():
    lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]
    
    sentiment_bow_ten_ks[ticker] = {
        sentiment: get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
        for sentiment in sentiments}


project_helper.print_ten_k_data([sentiment_bow_ten_ks[example_ticker]], sentiments)


# # Jaccard Similarity

# In[ ]:


"""
Using the bag of words, let's calculate the jaccard similarity on the bag of words and plot it over time. 
Implement get_jaccard_similarity to return the jaccard similarities between each tick in time. Since the input,
bag_of_words_matrix, is a bag of words for each time period in order, you just need to compute the jaccard 
similarities for each neighboring bag of words. Make sure to turn the bag of words into a boolean array when
calculating the jaccard similarity.
"""


# In[40]:


def get_jaccard_similarity(bag_of_words_matrix):
    """
    Get jaccard similarities for neighboring documents

    Parameters
    ----------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    jaccard_similarities : list of float
        Jaccard similarities for neighboring documents
    """
    
    # TODO: Implement
    jaccard_similarities = []
    bag_of_words_matrix = np.array(bag_of_words_matrix, dtype=bool)
    
    for i in range(len(bag_of_words_matrix)-1):
            u = bag_of_words_matrix[i]
            v = bag_of_words_matrix[i+1]
            jaccard_similarities.append(jaccard_score(u,v))    
    
    return jaccard_similarities


# In[ ]:


"""
Using the get_jaccard_similarity function, let's plot the similarities over time.
"""


# In[41]:


# Get dates for the universe
file_dates = {
    ticker: [ten_k['file_date'] for ten_k in ten_ks]
    for ticker, ten_ks in ten_ks_by_ticker.items()}  

jaccard_similarities = {
    ticker: {
        sentiment_name: get_jaccard_similarity(sentiment_values)
        for sentiment_name, sentiment_values in ten_k_sentiments.items()}
    for ticker, ten_k_sentiments in sentiment_bow_ten_ks.items()}


project_helper.plot_similarities(
    [jaccard_similarities[example_ticker][sentiment] for sentiment in sentiments],
    file_dates[example_ticker][1:],
    'Jaccard Similarities for {} Sentiment'.format(example_ticker),
    sentiments)


# # TFIDF

# In[ ]:


"""
Using the sentiment word lists, let's generate sentiment TFIDF from the 10-k documents. Implement get_tfidf to 
generate TFIDF from each document, using sentiment words as the terms. You can ignore words that are not in 
sentiment_words.
"""


# In[43]:


def get_tfidf(sentiment_words, docs):
    """
    Generate TFIDF values from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    
    # TODO: Implement
    vec = TfidfVectorizer(vocabulary=sentiment_words)
    tfidf = vec.fit_transform(docs)
    
    return tfidf.toarray()


# In[45]:


sentiment_tfidf_ten_ks = {}

for ticker, ten_ks in ten_ks_by_ticker.items():
    lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]
    
    sentiment_tfidf_ten_ks[ticker] = {
        sentiment: get_tfidf(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
        for sentiment in sentiments}

    
project_helper.print_ten_k_data([sentiment_tfidf_ten_ks[example_ticker]], sentiments)


# # Cosine Similarity

# In[ ]:


"""
Using the TFIDF values, we'll calculate the cosine similarity and plot it over time. Implement 
get_cosine_similarity to return the cosine similarities between each tick in time. Since the input, tfidf_matrix, 
is a TFIDF vector for each time period in order, you just need to computer the cosine similarities for each 
neighboring vector.
"""


# In[47]:


def get_cosine_similarity(tfidf_matrix):
    """
    Get cosine similarities for each neighboring TFIDF vector/document

    Parameters
    ----------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    cosine_similarities : list of float
        Cosine similarities for neighboring documents
    """
    
    # TODO: Implement
    cosine_similarities = []    
    
    for i in range(len(tfidf_matrix)-1):
        cosine_similarities.append(cosine_similarity(tfidf_matrix[i].reshape(1, -1),tfidf_matrix[i+1].reshape(1, -1))[0,0])
    
    return cosine_similarities


# In[48]:


cosine_similarities = {
    ticker: {
        sentiment_name: get_cosine_similarity(sentiment_values)
        for sentiment_name, sentiment_values in ten_k_sentiments.items()}
    for ticker, ten_k_sentiments in sentiment_tfidf_ten_ks.items()}


project_helper.plot_similarities(
    [cosine_similarities[example_ticker][sentiment] for sentiment in sentiments],
    file_dates[example_ticker][1:],
    'Cosine Similarities for {} Sentiment'.format(example_ticker),
    sentiments)


# # Price Data

# In[ ]:


"""
Let's evaluate the alpha factors. For this section, we'll just be looking at the cosine similarities, but it can 
be applied to the jaccard similarities as well.
"""


# In[50]:


pricing = pd.read_csv('./quotemedia.csv', parse_dates=['date'])
pricing = pricing.pivot(index='date', columns='ticker', values='adj_close')

pricing


# # Dict to DataFrame

# In[51]:


cosine_similarities_df_dict = {'date': [], 'ticker': [], 'sentiment': [], 'value': []}


for ticker, ten_k_sentiments in cosine_similarities.items():
    for sentiment_name, sentiment_values in ten_k_sentiments.items():
        for sentiment_values, sentiment_value in enumerate(sentiment_values):
            cosine_similarities_df_dict['ticker'].append(ticker)
            cosine_similarities_df_dict['sentiment'].append(sentiment_name)
            cosine_similarities_df_dict['value'].append(sentiment_value)
            cosine_similarities_df_dict['date'].append(file_dates[ticker][1:][sentiment_values])

cosine_similarities_df = pd.DataFrame(cosine_similarities_df_dict)
cosine_similarities_df['date'] = pd.DatetimeIndex(cosine_similarities_df['date']).year
cosine_similarities_df['date'] = pd.to_datetime(cosine_similarities_df['date'], format='%Y')


cosine_similarities_df.head()


# # Alphalens Format

# In[ ]:


"""
In order to use a lot of the alphalens functions, we need to aligned the indices and convert the time to unix
timestamp. In this next cell, we'll do just that.
"""


# In[67]:


import alphalens as al


factor_data = {}
skipped_sentiments = []
print(sentiments)

for sentiment in sentiments:
    cs_df = cosine_similarities_df[(cosine_similarities_df['sentiment'] == sentiment)]
    cs_df = cs_df.pivot(index='date', columns='ticker', values='value')
    
    try:
        data = al.utils.get_clean_factor_and_forward_returns(cs_df.stack(), pricing.loc[cs_df.index], quantiles=5, bins=None, periods=[1])
        factor_data[sentiment] = data
    except:
        skipped_sentiments.append(sentiment)        

if skipped_sentiments:
    print('\nSkipped the following sentiments:\n{}'.format('\n'.join(skipped_sentiments)))
factor_data[sentiments[0]].head()


# In[66]:


factor: data.set_index(pd.MultiIndex.from_tuples(
    [(x.timestamp(), y) for x, y in data.index.values],
    names=['date', 'asset']))
for factor, data in factor_data.items()}


# In[ ]:


ls_factor_returns = pd.DataFrame()

for factor_name, data in factor_data.items():
    ls_factor_returns[factor_name] = al.performance.factor_returns(data).iloc[:, 0]

(1 + ls_factor_returns).cumprod().plot()


# # References

# - https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92

# In[ ]:




