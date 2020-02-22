from nltk.stem import WordNetLemmatizer

def tfisf_scorer(sentences):
    """
    This method accepts a list of sentences and calculates the tf_isf scores for each of the sentence.
    The tf_isf algorithm is an adaption of the tf-idf algorithm that calculates the product of  frequency of a term
    (word) the inverse frequency of the text in the entire corpus. The corpus in the case of the tf_isf is the 
    set of sentences rather than the corpus of documents.
    
    1. Input : List of sentences 
    2. Output : List of sentence index with their respective tf_isf scores
    """
    sentences_score = []
    for i in range(len(sentences)-1):
        sentence_score = 0
        for word in sentences[i]:
            tf_score =  calculate_tf_score(word,sentence[i])
            idf_score = calculate_isf_score(word,sentences)
            w_score = tf_score * idf_score
            sentence_score += w_score
        sentence_score.append((i,sentence_score))
    return sentence_score

def calculate_tf_score(target_word, sentence):
        """
        Method calculates the text frequency score for a given word in a sentence
        1. Input : target_word and the sentence considered
        2. Output : tf_score of the given word. 
        """
        
    
