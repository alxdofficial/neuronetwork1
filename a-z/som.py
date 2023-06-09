import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# to find potential fraud, we want to identify users who are unique and different from others.
# so we need mean node distance

dataset = pd.read_csv("som_data/Credit_Card_Applications.csv")
x = dataset.iloc[:, : -1]
y = dataset.iloc[:, -1] # not used for training, just visualization

# scale features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)

# build the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
# 10 by 10 grid of neurons, inputlen is num columns (including the user id because we want to know which user it is
# later), sigma is neuron influence radius

# train som
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

# visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
# distance_map() returns matrix of nodal distances for every node
pcolor(som.distance_map().T)  # .T gives transpose of matrix
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
# list potential fraud accounts
for i, matrix_i in enumerate(x):
    # here i will be index into an array generated by enumerate, and matrix_i will be the row at that index
    winning_node = som.winner(matrix_i)
    plot(winning_node[0] + 0.5, winning_node[1] + 0.5, markeredgecolor=markers[y[i]], markerfacecolor=colors[y[i]],
         markersize=10, markeredgewidth=2)  # the + 0.5 puts it in the center
    # w[0] is x coordinate of winning node, w[1] is y coord

show()

mappings = som.win_map(x)
