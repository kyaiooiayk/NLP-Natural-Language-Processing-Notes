import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def document_term_matrix(corpus, vectorizer):
    '''
    Input: a cleaned corpus and vectorizer
    Output: a document-term matrix
    '''
    vec = vectorizer(tokenizer=lambda doc:doc, lowercase=False, min_df=2, max_df=0.5)
    matrix = vec.fit_transform(corpus).toarray()
        
    return vec, matrix

def word_frequency(vec, matrix):
    '''
    Input: a document-term matrix and an instantiated vectorizer
    Output: a sorted list of words and their frequencies
    '''
    sum_words = matrix.sum(axis=0)
    words_freq = [(word, sum_words[idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x:x[1], reverse=True)
    
    return words_freq

def topic_model(vec, matrix, model, num_topics, num_words):
    '''
    Input: an instantiated vectorizer, document-term matrix, type of topic model, number of topics, and number of words is each topic
    Output: a list of lists containing topic words
    '''
    ## Creates an instance of an NMF or LDA model
    if model == NMF:
        model = model(num_topics)
    elif model == LatentDirichletAllocation:
        model = model(n_components=num_topics)
        
    ## Fit_transform (matrix factorization for NMF) the doc_word matrix to get doc_topic and topic_word matrices
    doc_topic = model.fit_transform(matrix)
    topic_word = model.components_
    
    ## Retrieves the top words in each topic
    words = vec.get_feature_names()
    t_model = topic_word.argsort(axis=1)[:, -1:-(num_words+1):-1]
    top_topic_words = [[words[i] for i in topic] for topic in t_model]
        
    return top_topic_words, doc_topic

def kmeans_clustering(matrix, n_clusters):
    '''
    input: a document-term matrix and number of clusters in kmeans model
    output: a kmeans plot showing n_clusters
    '''
    ## Dimensionality reduction using PCA
    pca = PCA(n_components=2, random_state=1)

    matrix_centered = matrix - matrix.mean(axis=0)
    pca_model = pca.fit_transform(matrix_centered)
    
    ## Instantiate and fit a K-means clustering model
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_model = kmeans.fit(pca_model)
    
    ## Plot the clusters and return the cluster centers
    plt.scatter(pca_model[:,0], pca_model[:,1], c=kmeans_model.labels_, cmap='rainbow')
    
def silhouette_coeff_sse(matrix):
    '''
    input: a document-term matrix and number of clusters in kmeans model
    output: a kmeans plot and cluster centers
    '''
    ## Add silhouette scores and SSEs for each n_cluster to their respective lists
    SSEs = []
    Sil_coefs = []
    for k in range(2,20):
        pca = PCA(n_components=2, random_state=1)
        matrix_centered = matrix - matrix.mean(axis=0)
        pca_model = pca.fit_transform(matrix_centered)
        
        km = KMeans(n_clusters=k, random_state=1)
        km.fit(pca_model)
        labels = km.labels_
        Sil_coefs.append(silhouette_score(pca_model, labels, metric='euclidean'))
        SSEs.append(km.inertia_) 
        
    ## Create elbow plots for both 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), sharex=True)
    k_clusters = range(2,20)
    ax1.plot(k_clusters, Sil_coefs)
    ax1.set_xlabel('number of clusters')
    ax1.set_ylabel('silhouette coefficient')

    # plot here on ax2
    ax2.plot(k_clusters, SSEs)
    ax2.set_xlabel('number of clusters')
    ax2.set_ylabel('SSE');