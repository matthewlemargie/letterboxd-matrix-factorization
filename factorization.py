import torch
import torch.nn as nn
from torch.nn import Embedding

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super().__init__()
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.movie_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        users, movies = data[:,0], data[:,1]
        return (self.user_factors(users)*self.movie_factors(movies)).sum(1)

    def predict(self, data):
        return self.forward(data)
