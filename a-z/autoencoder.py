import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# import dataset
# We won't be using this dataset.
movies = pd.read_csv('boltzman_data/ml-1m/movies.dat', sep ='::', header = None, engine ='python', encoding ='latin-1')
users = pd.read_csv('boltzman_data/ml-1m/users.dat', sep ='::', header = None, engine ='python', encoding ='latin-1')
ratings = pd.read_csv('boltzman_data/ml-1m/ratings.dat', sep ='::', header = None, engine ='python', encoding ='latin-1')
# prepare training set and test set
training_set = pd.read_csv('boltzman_data/ml-100k/u1.base', delimiter ='\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('boltzman_data/ml-100k/u1.test', delimiter ='\t')
test_set = np.array(test_set, dtype = 'int')
# get num movies and num users because we will create a matrix each for test and training set
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

# create matrix where rows are users and columns are ratings for each movie because our neural network expects it as
# input
def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)
# convert matrix into a torch tensor because neural networks expect a tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# create class stacked autoencoder SAE , who is child of pytorch module class
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # first fully connected layer
        self.fc1 = nn.Linear(nb_movies, 20)  # nb_movies input, 20 nodes in first hidden layer,
        # second fully connected layer
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
# lr is learning rate, weight decay is speed that learning rate decreases

# training and optimizations
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # mean squared error
    s = 0.
    # s is number of users that has rated at least 1 movie, this is to save memory
    for id_user in range(nb_users):
        # torch takes 2d vector as input for neural network, so we duplicate input layer into batch
        input = Variable(training_set[id_user]).unsqueeze(0)
        # create copy of input
        target = input.clone()
        if torch.sum(target.data > 0) > 0: # if rated at least 1 movie. torch sum counts data entries where predi = true
            # get our prediction
            output = sae(input)  # somehow the sae object calls SAE.forward
            target.require_grad = False  # dont compute gradient with respect to target
            output[target == 0] = 0
            # calculate loss
            loss = criterion(output, target)
            # coefficient to change nb of movies for computing mean squared error later,
            # because we are only looking at movies with non-zero ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # compute back propagation on our loss
            # loss is an object, it will contain the vector of our gradient direction
            loss.backward()

            train_loss += np.sqrt(loss.item() * mean_corrector)

            s += 1.
            # take a step toward the gradient direction
            optimizer.step()

    print('epoch: ' + str(epoch) + " loss: " + str(train_loss/s))

# test the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.

print('test loss: '+str(test_loss/s))
