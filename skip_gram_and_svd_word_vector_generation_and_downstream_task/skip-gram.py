
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.nn.utils.rnn import pad_sequence


import numpy as np

class SimpleWord2Vec:
    def __init__(self, sentences, vector_size=100, window=5, negative=5, learning_rate=0.01, epochs=10):
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self.weights_input = None
        self.weights_output = None

    def preprocess(self):
        all_words = [word for sentence in self.sentences for word in sentence]
        self.vocab_size = len(set(all_words))
        self.word_index = {word: idx for idx, word in enumerate(set(all_words))}
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        
        self.weights_input = np.random.rand(self.vocab_size, self.vector_size) - 0.5
        self.weights_output = np.random.rand(self.vector_size, self.vocab_size) - 0.5

    def train(self):
        for epoch in range(self.epochs):
            for sentence in self.sentences:
                sentence_indices = [self.word_index[word] for word in sentence]
                for center_word_pos, center_word_idx in enumerate(sentence_indices):
                    start = max(0, center_word_pos - self.window)
                    end = min(len(sentence), center_word_pos + self.window + 1)
                    context_indices = sentence_indices[start:center_word_pos] + sentence_indices[center_word_pos+1:end]
                    
                    for context_word_idx in context_indices:
                        self.gradient_descent(center_word_idx, context_word_idx, 1)
                    
                    for _ in range(self.negative):
                        negative_word_idx = np.random.randint(0, self.vocab_size)
                        if negative_word_idx not in context_indices:
                            self.gradient_descent(center_word_idx, negative_word_idx, 0)
            print(f"Epoch {epoch+1} complete")
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient_descent(self, center_word_idx, context_word_idx, label):
        center_word_vector = self.weights_input[center_word_idx]
        context_word_vector = self.weights_output[:, context_word_idx]
        score = np.dot(center_word_vector, context_word_vector)
        predicted = self.sigmoid(score)
        
        g = (predicted - label) * self.learning_rate
        gradient_center_word = g * context_word_vector
        gradient_context_word = g * center_word_vector
        
        self.weights_input[center_word_idx] -= gradient_center_word
        self.weights_output[:, context_word_idx] -= gradient_context_word

    def get_embedding(self, word):
        idx = self.word_index[word]
        return self.weights_input[idx]
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({
                "vector_size": self.vector_size,
                "window": self.window,
                "negative": self.negative,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "word_index": self.word_index,
                "index_word": self.index_word,
                "vocab_size": self.vocab_size,
                "weights_input": self.weights_input,
                "weights_output": self.weights_output,
            }, f)
    
    @classmethod
    def load_model(cls, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            model = cls(sentences =[],vector_size=data["vector_size"], window=data["window"], negative=data["negative"], 
                        learning_rate=data["learning_rate"], epochs=data["epochs"])
            model.word_index = data["word_index"]
            model.index_word = data["index_word"]
            model.vocab_size = data["vocab_size"]
            model.weights_input = data["weights_input"]
            model.weights_output = data["weights_output"]
            return model


train_df = pd.read_csv('train.csv')

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

train_df['Processed_Description'] = train_df['Description'].apply(preprocess_text)

sentences = train_df['Processed_Description'].tolist()

model1 = SimpleWord2Vec(sentences, vector_size=100, window=5, negative=5, learning_rate=0.01, epochs=10)
model1.preprocess()
model1.train()




embeddings = {word: model1.get_embedding(word) for word in model1.word_index}
import torch
torch.save(embeddings, 'skip-gram-word-vectors.pt')
