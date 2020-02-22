from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nlkt import pos_tag
from nlkt.tokenize import word_tokenize,sent_tokenize
import string
import operator
import math
from gensim.models import Doc2Vec, TaggedDocument
from sklearn.cluster import Birch
import os
import pikepdf


def summarizeDocument(document):
    sentences = sent_tokenize(documents)
    threshold = set_threshold(sentences) # done
    cleaned_sentences = preprocess(sentences) # done
    sents = cluster_sentences(cleaned_sentences) # done
    sentence_scores = tf_isf_scorer(sents) # done
    summary = generate_summary(sentences, sentence_scores, threshold)
    return summary



def set_threshold(sentences):
    return int(0.5 * len(sentences))


##########################################################
# 1.0 PREPROCESSING
def preprocess(sentences):
    white_list = ['Allah', 'God']
    sents = []
    for sent in sentences:
        sen = [word.lower() for word in word_tokenize(sen)]
        sen = [lemmatize(word) for word in sen]
        sen = pos_tag(sen)
        sen = [word for word in sen if word not in stopwords]
        sents.append(sen)
    return sents

def lemmatize(word):
    wnl = WordNetLemmatizer
    lemmed_words = []
    for word in word_tokenize(sentence):
        wrd = wnl.lemmatize(word)
        lemmed_words.append(wrd)
    return lemmed_words

def pos_tag(sentences):
    """
    takes a whole sentence as input and gets the Part-Of-Speech tag for each word in the sentence
    """
    acceptable_words = []
    for word in sentences:
        word_tag = pos_tag(word)
        tag = word_tag['tag']
        if (tag == "NP" or tag == "VP"):
            acceptable_words.append(word_tag['word'])
    return acceptable_words

##########################################################
# 2.0 CLUSTERING

def cluster_sentences(sentences):
    X = vectorize(sentences)
    bcl = Birch(branching_factor=10, n_clusters=None).fit(X) # the algorithm figures out the clusters
    clusters = bcl.predict(X)
    labels = bcl.labels_
    norm_X = normalize_vectors(X,labels)
    cluster_means = calculate_mean(norm_X,clusters)
    cluster_sentences = find_minimum_from_mean(cluster_means, norm_X)
    sents = vectors_to_sentences(cluster_sentences)
    print(sents)
    return sents

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

def find_minimum_from_mean():
    """
    This function computes the minimal distance between each element in X and the elements in Y
    cluster_means : A 2D array
    points : A 2D array consisting of cluster type and the corresponding value

    It returns an array which consist of point in X and the corresponding point in Y with the smallest distance the point in X

    """
    minimal_distances = []
    for clm in cluster_means:
        points_in_cluster = [point[1] for point in points if point[0] == clm[0]]
        minimal = points_in_cluster[0]
        for pt in points_in_cluster:
            diff_current = clm[1] - pt # maybe absolute value ??
            diff_minimal = clm[1] - minimal
            if (diff_current < diff_minimal):
                  minimal = pt
        minimal_distances.append(minimal)
    return minimal_distances


def vectors_to_sentences(top_sentences, sentence_vector, sentences):
    index = 0
    selected_sentences = []
    for v in sentence_vector:
        if v[1] in top_sentences:
            print(v[1])
            selected_sentences.append(sentences[index])
        index +=1
    return selected_sentences

def vectorize(sentences):
    words = [word for word in word_tokenize(sent) for sent in sent_tokenize(sentences)]
    X = [TaggedDocument(word,idnx) for idnx,word in enumerate(words)]
    vectorizer = Doc2Vec(X, size=10)

    sentence_vectors = [vectorizer.infer_vector(word_tokenize(sen), alpha=2,steps=400) for sen in sentences]
    return sentence_vectors

######################################################################
# 3. TF_ISF SCORER
def tf_isf_scorer(sentences):
    sentence_scores = []
    k = 0
    for sent in sentences:
        word_scores = 0
        for word in word_tokenize(sent):
            tf_score = tf_scorer(word,sent)
            idf_score = idf_scorer(word,sentences)
            score = tf_score * idf_score
            word_score += score
        sentence_scores.append((k,word_scores))
        k +=1
    return sentence_scores

def tf_scorer(word,sent):
    return word_tokenize(sent).count(word)

def idf_scorer(word,sents):
    count = 0
    for sent in sents:
        if word in sent:
            count +=1
    score = math.log(len(sents)/count)
    return score


def generate_summary(sentences,scores,threshold):
        ranked_scores = sorted(scores, key=lambda sc:sc[1], reverse=True)
        summary = []
        for sent_rank in ranked_scores:
            summary.append(sent_rank[0])
        return summary


##########################################################
# 4.0
def viz_clusters():
    return clusters
