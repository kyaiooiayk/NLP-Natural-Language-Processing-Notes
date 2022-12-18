"""
What? Test preprocessing utilities

Utilies include:
    [1] scatter_msft_df
    [2] scatterplot
    [3] document_term_matrix_df
    [4] top_ten_words
    [5] keyword_frequency
    [6] wordcloud
    [7] word2vec_plot
    [8] sentiment_analysis
    [9] sentiment_analysis_heatmap
    
Reference: https://mikechoi90.medium.com/investigating-microsofts-transformation-under-satya-nadella-f49083294c35
"""

# Import libraries
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scattertext as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pysentiment2 as ps
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from gensim.models import word2vec


def scatter_msft_df(corpus_ballmer, corpus_nadella):
    '''
    input: a list of lists containing cleaned transcript text
    output: a dataframe with text, CEO, and quarter (for creating a scattertext plot)
    '''
    ## Convert lists of tokens into lists of text strings
    corpus_text_ballmer = []
    for doc in corpus_ballmer:
        corpus_text_ballmer.append(' '.join(doc))
        
    corpus_text_nadella = []
    for doc in corpus_nadella:
        corpus_text_nadella.append(' '.join(doc))
        
    ## Create ceo and. quarter columns
    quarters_ballmer = ['msft_07q3', 'msft_07q4', 'msft_08q1', 'msft_08q2', 
                        'msft_08q3', 'msft_08q4', 'msft_09q1', 'msft_09q2', 
                        'msft_09q3', 'msft_09q4', 'msft_10q1', 'msft_10q2', 
                        'msft_10q3', 'msft_10q4', 'msft_11q1', 'msft_11q2', 
                        'msft_11q3', 'msft_11q4', 'msft_12q1', 'msft_12q2', 
                        'msft_12q3', 'msft_12q4', 'msft_13q1', 'msft_13q2', 
                        'msft_13q3', 'msft_13q4', 'msft_14q1', 'msft_14q2']
    quarters_nadella = ['msft_14q3', 'msft_14q4', 'msft_15q1', 'msft_15q2', 
                        'msft_15q3', 'msft_15q4', 'msft_16q1', 'msft_16q2', 
                        'msft_16q3', 'msft_16q4', 'msft_17q1', 'msft_17q2', 
                        'msft_17q3', 'msft_17q4', 'msft_18q1', 'msft_18q2', 
                        'msft_18q3', 'msft_18q4', 'msft_19q1', 'msft_19q2', 
                        'msft_19q3', 'msft_19q4', 'msft_20q1', 'msft_20q2', 
                        'msft_20q3', 'msft_20q4', 'msft_21q1', 'msft_21q2']
    
    df_ballmer_scatter = pd.DataFrame(corpus_text_ballmer, columns=['text'])
    df_ballmer_scatter['ceo'] = 'Ballmer'
    df_ballmer_scatter['quarter'] = quarters_ballmer
    
    df_nadella_scatter = pd.DataFrame(corpus_text_nadella, columns=['text'])
    df_nadella_scatter['ceo'] = 'Nadella'
    df_nadella_scatter['quarter'] = quarters_nadella
    
    ## Concatenate text, ceo, and quarter columns into a single dataframe
    df_msft_scatter = pd.concat([df_ballmer_scatter, df_nadella_scatter], axis=0).reset_index(drop=True)
    
    return df_msft_scatter


def scatterplot(df):
    '''
    input: a dataframe with text, CEO, and quarter 
    output: a scatterplot
    '''
    corpus = st.CorpusFromPandas(df, category_col='ceo', text_col='text',
                                 nlp=st.whitespace_nlp_with_sentences).build()
    
    html = st.produce_scattertext_explorer(
        corpus,
        category='Ballmer',
        category_name='Steve Ballmer Era',
        not_category_name='Satya Nadella Era',
        minimum_term_frequency=10,
        pmi_threshold_coefficient=5,
        width_in_pixels=1000,
        metadata=df['quarter'],
        )
    
    #open('../Charts/scattertext_demo.html', 'wb').write(html.encode('utf-8'));
    
def document_term_matrix_df(corpus, vectorizer):
    '''
    Input: a cleaned corpus and vectorizer
    Output: a document-term matrix in a dataframe
    '''
    vec = vectorizer(tokenizer=lambda doc:doc, lowercase=False, min_df=2)
    matrix = vec.fit_transform(corpus).toarray()
    df = pd.DataFrame(matrix, columns=vec.get_feature_names())
        
    return df

def top_ten_words(corpus, df):
    '''
    Input: a cleaned corpus and a document-term matrix in a dataframe
    Output: a dataframe showing top 10 most frequently used words in transcripts over time
    '''
    ## For each transcript, top 10 most frequent words are appended to a list
    top10_words = []
    for i in range(len(corpus)):
        top10_words.append(list(df.iloc[i].sort_values(ascending=False).head(10).index))
        
    df_top10_words = pd.DataFrame(top10_words).T
    
    ## Create column headers - multilevel headers of fiscal years and quarters
    fy_list = ['FY2007','FY2008','FY2009','FY2010',
               'FY2011','FY2012','FY2013','FY2014',
               'FY2015','FY2016','FY2017','FY2018',
               'FY2019','FY2020','FY2021']
    fy_quarters = ['Q1','Q2','Q3','Q4']
    
    fy_quarter_pair = []
    for year in fy_list:
        if year=='FY2007':
            fy_quarter_pair.extend([('FY2007', fy_quarters[2]), ('FY2007', fy_quarters[3])])
        elif year=='FY2021':
            fy_quarter_pair.extend([('FY2021', fy_quarters[0]), ('FY2021', fy_quarters[1])])
        else:
            for quarter in fy_quarters:
                fy_quarter_pair.append((year, quarter))
                
    df_top10_words.columns = pd.MultiIndex.from_tuples(fy_quarter_pair)
    
    return df_top10_words

def keyword_frequency(df, keyword_list):
    '''
    input: a list of keywords that appear in transcripts and a document-term matrix in a dataframe
    output: a heatmap showing keyword frequency over time
    '''
    ## Put in the keyword list as a mask to filter the document-term matrix dataframe
    df_key_freq = df.T.loc[keyword_list]
    
    ## Create column headers - multilevel headers of fiscal years and quarters
    fy_list = ['FY2007','FY2008','FY2009','FY2010',
               'FY2011','FY2012','FY2013','FY2014',
               'FY2015','FY2016','FY2017','FY2018',
               'FY2019','FY2020','FY2021']
    fy_quarters = ['Q1','Q2','Q3','Q4']
    
    fy_quarter_pair = []
    for year in fy_list:
        if year=='FY2007':
            fy_quarter_pair.extend([('FY2007', fy_quarters[2]), ('FY2007', fy_quarters[3])])
        elif year=='FY2021':
            fy_quarter_pair.extend([('FY2021', fy_quarters[0]), ('FY2021', fy_quarters[1])])
        else:
            for quarter in fy_quarters:
                fy_quarter_pair.append((year, quarter))
    
    df_key_freq.columns = pd.MultiIndex.from_tuples(fy_quarter_pair)
    df_key_freq = df_key_freq.div(df_key_freq.max(axis=1), axis=0)
    
    ## Create a heatmap showing keyword frequency over time
    fig, ax = plt.subplots(1, 1, figsize = (15, 6))
    sns.heatmap(df_key_freq, cmap='Blues', annot=False)
    ax.set_ylabel('Keywords', fontsize=15)    
    ax.set_xlabel('Fiscal Period', fontsize=15)
    plt.tight_layout()
    
def wordcloud(corpus):
    '''
    Input: a cleaned corpus
    Output: a Wordcloud visualization showing most frequently represented words
    '''
    ## Create a document-term matrix    
    vec = CountVectorizer(tokenizer=lambda doc:doc, lowercase=False, min_df=2, max_df=0.5)
    matrix = vec.fit_transform(corpus).toarray()
    
    ## A list of words sorted by word frequency in the text
    sum_words = matrix.sum(axis=0)
    words_freq = [(word, sum_words[idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x:x[1], reverse=True)
    
    ## Create a wordcloud viz
    wordcloud = WordCloud(width=400, height=330, max_words=150,colormap="Dark2")
    
    wordcloud.generate_from_frequencies(dict(words_freq))

    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")    
    
def word2vec_plot(corpus, min_count, window):
    '''
    Input: Cleaned dataframe 
    Output: A t-SNE plot showing clusters based on Word2Vec embeddings
    '''    
    ## Initialize a Word2Vec model and set parameters
    model = word2vec.Word2Vec(corpus, min_count=min_count, window=window) #, ns_exponent = -10)
    
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    ## Perform dimensionality reduction using TSNE
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    ## Create a plot using matplotlib
    plt.figure(figsize=(16, 8)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        
    plt.tight_layout()
    #plt.savefig('../Charts/word2vec_tsne_plot', dpi=600)
    
def sentiment_analysis(corpus, lmfsd):
    '''
    input: a cleaned corpus and a csv file containing Loughran & McDonald Financial Sentiment Dictionaries
    output: a dataframe of frequencies of sentiment words (positive, negative, uncertain) over time
    '''
    positive = lmfsd[lmfsd["Positive"]>0]
    negative = lmfsd[lmfsd["Negative"]>0]
    uncertainty = lmfsd[lmfsd["Uncertainty"]>0]
    
    pos_words = [word.lower() for word in positive["Word"].tolist()]
    neg_words = [word.lower() for word in negative["Word"].tolist()]
    uncert_words = [word.lower() for word in uncertainty["Word"].tolist()]
    
    corpus_pos_words = {}
    corpus_neg_words = {}
    corpus_uncert_words = {}
    
    for tscript in corpus.keys():
        corpus_pos_words[tscript] = [word for word in corpus[tscript] if word in pos_words]
        corpus_neg_words[tscript] = [word for word in corpus[tscript] if word in neg_words]
        corpus_uncert_words[tscript] = [word for word in corpus[tscript] if word in uncert_words]
        
    corpus_pos_count = [len(doc) for doc in list(corpus_pos_words.values())]
    corpus_neg_count = [len(doc) for doc in list(corpus_neg_words.values())]
    corpus_uncert_count = [len(doc) for doc in list(corpus_uncert_words.values())]
    
    df_sentiment_count = pd.DataFrame({'Positive': corpus_pos_count, 
                                   'Negative': corpus_neg_count, 
                                   'Uncertainty': corpus_uncert_count, 
                                  }, index=corpus.keys()).T
    
    fy_list = ['FY2007','FY2008','FY2009','FY2010',
               'FY2011','FY2012','FY2013','FY2014',
               'FY2015','FY2016','FY2017','FY2018',
               'FY2019','FY2020','FY2021']
    fy_quarters = ['Q1','Q2','Q3','Q4']

    fy_quarter_pair = []
    for year in fy_list:
        if year=='FY2007':
            fy_quarter_pair.extend([('FY2007', fy_quarters[2]), ('FY2007', fy_quarters[3])])
        elif year=='FY2021':
            fy_quarter_pair.extend([('FY2021', fy_quarters[0]), ('FY2021', fy_quarters[1])])
        else:
            for quarter in fy_quarters:
                fy_quarter_pair.append((year, quarter))
                
    df_sentiment_count.columns = pd.MultiIndex.from_tuples(fy_quarter_pair)
    
    return df_sentiment_count

def sentiment_analysis_heatmap(df):
    
    '''
    input: a dataframe of frequencies of sentiment words (positive, negative, uncertain) over time
    output: a heatmap of frequencies of sentiment words (positive, negative, uncertain) over time
    '''
    df = df.div(df.max(axis=1), axis=0)    
    fig, ax = plt.subplots(1, 1, figsize = (15, 5), dpi=600)
    sns.heatmap(df, cmap='Greens', annot=False)
    ax.set_ylabel('Sentiment', fontsize=15)    
    ax.set_xlabel('Fiscal Period', fontsize=15)
    ax.set_title("Change in Sentiment in the Q&A of Microsoft's Earnings Calls over Time", fontsize=18)
    #plt.tight_layout()
    #print("here")
    