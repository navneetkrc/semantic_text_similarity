import pandas as pd
import numpy as np
import random, sys, os
from nltk.tokenize import RegexpTokenizer

PATH="../real_not_real_kaggle/data/"



# !ls ../input/glove-global-vectors-for-word-representation
glove_dir = f'{PATH}/' # This is the folder with the dataset

glove_embedding = {} # We create a dictionary of word -> embedding
# f = open(os.path.join(glove_dir, 'glove.6B.50d.txt')) # Open file
f = open(f'{PATH}glove.6B.50d.txt') # Open file

# In the dataset, each line represents a new word embedding
# The line starts with the word and the embedding values follow
count=0
try:
    for line in f:
        values = line.split()
        word = values[0] # The first value is the word, the rest are the values of the embedding
        embedding = np.asarray(values[1:], dtype='float32') # Load embedding
        glove_embedding[word] = embedding # Add embedding to our embedding dictionary
except:
    count=count+1
f.close()

print('Found %s word vectors.' % len(glove_embedding), count)



def tokenizeText(df, text_column):
    tokenizer = RegexpTokenizer(r'\w+')
    df["tokens"] = df[text_column].apply(tokenizer.tokenize)
    return df



def get_average_word2vec(tokens_list, pretrained_word_vector, generate_missing=False, num_dims = 50):
    if len(tokens_list)<1:
        return np.zeros(num_dims)
    if generate_missing:
        vectorized = [pretrained_word_vector[word] if word in pretrained_word_vector else np.random.rand(num_dims) for word in tokens_list]
    else:
        vectorized = [pretrained_word_vector[word] if word in pretrained_word_vector else np.zeros(num_dims) for word in tokens_list]
#        print(np.array(vectorized).shape)
    length = len(vectorized)
#    print(length)
    summed = np.sum(vectorized, axis=0)
#    print(np.array(summed).shape)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(df, token_column, generate_missing=False):
    df = tokenizeText(df, "text")

    embeddings = df[token_column].apply(lambda x: get_average_word2vec(x, glove_embedding, generate_missing=generate_missing))
    return list(embeddings)





