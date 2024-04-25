import pickle
import nltk
nltk.download('punkt')

import pandas as pd
from collections import Counter
from scipy.sparse import lil_matrix
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy  as np


from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def generate_embeddings(window_size):
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        return tokens
    
    train_df = pd.read_csv('train.csv')
    train_df['Processed_Description'] = train_df['Description'].apply(preprocess_text)
    
    vocab = Counter()
    for tokens in train_df['Processed_Description']:
        vocab.update(tokens)
    vocab = {word: i for i, word in enumerate(vocab)}
    
    co_occurrence_matrix = lil_matrix((len(vocab), len(vocab)), dtype='float32')
    
    def count_cooccurrences(tokens, window_size):
        co_occurrence_counts = Counter()
        for i, token in enumerate(tokens):
            if token in vocab:
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(tokens))
                for j in range(start, end):
                    if i != j and tokens[j] in vocab:
                        ordered_pair = tuple(sorted([vocab[token], vocab[tokens[j]]]))
                        co_occurrence_counts[ordered_pair] += 1
        return co_occurrence_counts
    
    for tokens in train_df['Processed_Description']:
        co_occurrences = count_cooccurrences(tokens, window_size)
        for (word_i, word_j), count in co_occurrences.items():
            co_occurrence_matrix[word_i, word_j] += count
    
    co_occurrence_matrix = co_occurrence_matrix.tocsr()
    
    n_components = 100
    svd = TruncatedSVD(n_components=n_components)
    word_embeddings = svd.fit_transform(co_occurrence_matrix)
    
    embeddings = {word: word_embeddings[word_index] for word, word_index in vocab.items()}
    
    return embeddings

# Example usage:
window_size = 2
embeddings = generate_embeddings(window_size)
import torch
torch.save(embeddings, 'svd-word-vectors.pt')
