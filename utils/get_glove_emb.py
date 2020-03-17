import pandas as pd
import numpy as np
import random, sys, os
from nltk.tokenize import RegexpTokenizer
import numpy as np


PATH="/content/semantic_text_similarity/"
word_index=pd.read_pickle(f"{PATH}glove.840B.300d.pkl")

def avg_feature_vector(sentence, num_features, word_index):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in word_index:
            n_words += 1
            feature_vec = np.add(feature_vec, word_index[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_glove_embeddings(df):
    df["emb"] = df["description"].apply(lambda x: avg_feature_vector(x, 300, word_index)) 
    
    return df