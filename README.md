## Semantic Text Similarity
End-to-end searching pipeline for finding semantic text similarity. By using GloVe embeddings we can perform semantic search on a large corpus of text. [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library that is used to quickly search for embeddings of multi-dimensional documents that are similar to each other.

## Setup:
- Install FAISS.
- Download GloVe vectors.
- Install Pandas, Numpy.
- Done!

## Usage:
- train.pkl is the data on which we want to perform the search operation.
- test.pkl is the data which we will use as the query set.
- similar_git.py is the file containing code for text searching using FAISS.
- Notebook similar_test.ipynb is main file from which we perform the search.
- We can also use any embeddings other than GloVe. Just update the vector column named "emb" in train and test data file.
- Similarity results are returned in a dataframe. 

## Credits and Downloads:
- Data is taken from [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview) Kaggle competition. Download data [here](https://www.kaggle.com/c/nlp-getting-started/data). 
- Download 300d GloVe embeddings [here](https://www.kaggle.com/authman/pickled-glove840b300d-for-10sec-loading). 
