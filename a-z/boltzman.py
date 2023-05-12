import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# import the data
movies = pd.read_csv("boltzman_data/ml-1m/movies.dat", sep="::", header=None, engine="python", encoding="latin-1")
users = pd.read_csv("boltzman_data/ml-1m/users.dat", sep="::", header=None, engine="python", encoding="latin-1")
ratings = pd.read_csv("boltzman_data/ml-1m/ratings.dat", sep="::", header=None, engine="python", encoding="latin-1")
# in ratings, col 0 = userid, col 1 = movie id, col 2 = rating, col 3 = timestamp

# we will use ml 100k u1 as training set
training_set = pd.read_csv("boltzman_data/ml-100k/u1.base", delimiter="\t")
# need convert to array for pytorch, before it is dataframe
training_set = np.array(training_set, dtype='int')  # since all u1 is integers, set d type to int to get matrix of ints

# prepare test set
test_set = pd.read_csv("boltzman_data/ml-100k/u1.test", delimiter="\t")
test_set = np.array(test_set, dtype='int')

# create a matrix each for train and test where rows are users and cols are movies and each cell is rating
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# rbm expects n x m matrix as input
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# convert into torch tensors, a tensor is just a multidimensional matrix
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# convert to binary rating for simplicity. map all 0 ratings to -1 because -1 means no rating
training_set[training_set == 0] = -1
# 1 2 means not like, 3 and above means like
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
# do the same for test set
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[training_set >= 3] = 1

class RMB():
    def __int__(self, nv, nh):
        # randnormal creates a nh x nv sized tensor with rand(0,1) representing weights of connections
        self.W = torch.randn(nh, nv)
        # bias A of P(hidden | visible)
        self.a = torch.randn(1, nh)
        # bias B of P(visible | hidden)
        self.a = torch.randn(1, nv)

    # x is visible layer, return the vector of h
    def sample_h(self, x):
        # a neuron in rbm node is activated by the sum of weight i times input i
        wx = torch.mm(x , self.W.t())
        # we need to do transpose on w for math reasons
        activation = wx + self.a.expand_as(wx)
        ph_given_v = torch.sigmoid(activation)
        # return the vector of probabilities phgivenv
        # return a bernouli sample of hidden nodes, where the bernoulli generates rand(0,1) and if random number is
        # larger than ph_given_v[i]. sample[i] = 1, else sample[i] = 0
        return ph_given_v, torch.bernoulli(ph_given_v)

    # y is vector of values of hidden nodes
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

