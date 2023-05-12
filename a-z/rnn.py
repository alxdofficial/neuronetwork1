import keras.layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import only training set for now. 1 we do so using pandas open. 1.5 select needed columns.
# 2 we convert dataframe to numpy array because keras requires numpy array as input to neural network
dataset_train = pd.read_csv("rnn_data/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
# here we just want index 1, but 1:2 in python doesnt include 2. this is needed to make sure its a numpy multi
# dimensional array
#
# we always need to feature scale, in rnn normalization is best
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

# # we need to create a datastructure with 60 time steps and 1 output. rnn will look at 60 time steps ago to solve for
# # any one neuron's output
# # we also need to create a structure that contains the correct answer for any neuron at time t
# # so xtrain is (each day is time t, get all stock prices t-59 days ago). y train is (stock price t + 1)
# # x y train is actually 2d array
# x_train = []
# y_train = []
# for i in range(60, 1258):
#     x_train.append(training_set_scaled[i-60:i, 0])
#     # [i-60:i, 0] the 0 is for the first column
#     y_train.append(training_set_scaled[i, 0])
# # turn x y train into numpy array for keras needs that format
# x_train, y_train = np.array(x_train), np.array(y_train)
#
# # we can add even more dimensions to this structure to represent other indicators like volume close or other related
# # stocks. first we need to reshape the array for a third dimension
# # np.reshape(matrix to reshape, new shape class)
# # new shape class(# of data entries, time steps, indicators at each timestep)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
# print(x_train.shape[0])
#
# # build the rnn
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
#
# # continuous output, so regression not classification
# regressor = Sequential()
# # add lstm layer 1. LSTM class takes (units = # of output neurons, input shape =  (# of timesteps, # of indicators))
# regressor.add(keras.layers.LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
# # add dropout regularization
# regressor.add(Dropout(0.2))  # dropout 20% of lstm is classic number, aka ignore 20% of the neurons to avoid overfitting
#
# # repeat for 3 layers. only need to input shape for first layer
# regressor.add(LSTM(units = 100, return_sequences = True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 100, return_sequences = True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 100, return_sequences = True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 100))  # dont need return sequence for final lstm layer
# regressor.add(Dropout(0.2))
#
# # add final output layer
# regressor.add(keras.layers.Dense(units=1))
#
# # compile
# regressor.compile(optimizer="adam", loss="mean_squared_error")
#
# # train
# regressor.fit(x_train, y_train, epochs = 250, batch_size = 32)
# regressor.save("save rnn 1")
#

#
#

# now we can try a prediction

regressor = keras.models.load_model("save rnn 1")

# 1. since we trained the model to use all the datat since t-60, we need to supply it with 60 days of prices
# 2. we need to scale our input, its a good idea to create a new datastructure as to not modify our actual data
dataset_test = pd.read_csv('rnn_data/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)   # axis=0 means concat vertical rows
# we want jan 3 2017
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
# -1,1 means all the rows, 1 column
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_prices = regressor.predict(X_test)
# this prediction would be feature-scaled, so we need to inverse scale it
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

# lets visualize the prediction
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_prices, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()