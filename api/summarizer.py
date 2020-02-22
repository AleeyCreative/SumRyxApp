## Extractive Text Summarization Using Hierarchical Clustering
## by  Abdul Aziz Baba Ali

# importing dependencies
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
import string
import operator
import math
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import Birch
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pikepdf
from whitelist_check import *


def preprocess(words):
    stp = stopwords.words('english')
    words = whitelist_check(words)
    words = [word.lower() for word in words]
    words = [lemmatize(word) for word in words]
    words = [word for word in words if word not in stp]
    words = [word for word in words if word not in string.punctuation]
    return words

def lemmatize(word):
    wnl = WordNetLemmatizer()
    lemmed = wnl.lemmatize(word)
    return lemmed

def tagger(sentence):
    """
    takes a whole sentence as input and gets the Part-Of-Speech tag for each word in the sentence
    """
    acceptable_words = []
    word_tags = pos_tag(sentence)
    print(word_tags)
    for word_tag in word_tags:
        if (word_tag[1] == "NP" or word_tag[1] == "VP"):
            acceptable_words.append(word_tag[0])
    return acceptable_words

def vectorize(document):
    words = word_tokenize(document)
    words = preprocess(words)
    X = [TaggedDocument(word,[idnx]) for idnx,word in enumerate(words)]
    vectorizer = Doc2Vec(X, size=10)
    sentence_vectors = [vectorizer.infer_vector(word_tokenize(sen), alpha=2,steps=400) for sen in sent_tokenize(document)]
    return sentence_vectors

def set_threshold(document):
    no_sen = len(sent_tokenize(document)) # the number of sentences in the document
    if(no_sen <= 50):
        return 0.4
    if(no_sen > 50 and no_sen <= 100):
        return 0.7
    if(n0_sen > 100  and no_sen <= 150):
        return 0.9
    if(no_sen > 150):
        return 0.9

def calculate_mean(X):
    clusters = set([sent[0] for sent in X ])
    cluster_mean = []
    for cluster in clusters:
        cluster_value = 0
        count = 0
        for i,element in enumerate(X):
            if (cluster == element[0]):
                cluster_value += element[1]
                count += 1
        mean_cluster_value = cluster_value/count
        cluster_mean.append([cluster,mean_cluster_value])
    print("Cluster Means:")
    print(cluster_mean)
    return cluster_mean

def find_minimum_from_mean(cluster_means, vectorized):
    """
    This function computes the minimal distance between each element in X and the elements in Y
    cluster_means : A 2D array
    points : A 2D array consisting of cluster type and the corresponding value

    It returns an array which consist of point in X and the corresponding point in Y with the smallest distance the point in X

    """
    minimal_distances = []
    for clm in cluster_means:
        points_in_cluster = [v[1] for v in vectorized if v[0] == clm[0]]
        minimal = points_in_cluster[0]
        for pt in points_in_cluster:
            diff_current = abs(clm[1] - pt)
            diff_minimal = clm[1] - minimal
            if (diff_current < diff_minimal):
                  minimal = pt
        minimal_distances.append((clm[0],minimal))
    print("Minimal", minimal_distances)
    return minimal_distances

def normalize_vector(v, labels):
    vectorized = []
    for i in range(len(v)):
        mean = abs(sum(v[i]/len(v[i])))
        vectorized.append((labels[i],mean))
    return vectorized

def vectors_to_sentences(clustered_sentences, sentence_vector, sentences):
    index = 0
    selected_sentences = []
    top_sentences = [s[1] for s in clustered_sentences ]
    for v in sentence_vector:
        if v[1] in top_sentences:
            print(v[1])
            selected_sentences.append((index,sentences[index]))
        index +=1
    return selected_sentences

def cluster_and_extract_sentences(document):
    X = vectorize(document)
    th = set_threshold(document)
    bcl = Birch(branching_factor=10, n_clusters=None, threshold=th).fit(X) # the algorithm figures out the clusters
    clusters = bcl.predict(X)
    labels = bcl.labels_
    norm_X = normalize_vector(X, labels)
    # viz_clusters(norm_X,labels) # visualization before finding the mean
    cluster_means = calculate_mean(norm_X)
    cluster_sentences = find_minimum_from_mean(cluster_means, norm_X)
    # viz_clusters_after(cluster_sentences, set(labels)) # visualization after finding the closest to mean
    sents = vectors_to_sentences(cluster_sentences, norm_X, sent_tokenize(document))
    # sentence_data = pd.DataFrame(sentence_data, columns = ['Sentence', 'Vector_Value', ' Cluster_ID', 'Cluster_Mean', 'Closest_Sentence_Vector'])
    return sents


def summarizeDocument(document, title):
    """
    The function takes a document and returns its summary
    """
    sentences = sent_tokenize(document)
    indexed_sentences = cluster_and_extract_sentences(document)
    summary = [sent[1] for sent in indexed_sentences]
    summary =  " ".join(summary)
    return {'title':title, 'content': summary}


