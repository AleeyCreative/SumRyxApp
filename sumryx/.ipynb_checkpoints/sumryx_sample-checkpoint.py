from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nlkt import pos_tag
from nlkt.tokenize import word_tokenize,sent_tokenize
import string
import operator
from gensim.models import Doc2Vec
from sklearn.cluster import Birch
import os
import pikepdf



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
        
def vectorize(sentences):
    
    return sentence_vector

def tf_isf_scorer(sentences):
    
    return scores