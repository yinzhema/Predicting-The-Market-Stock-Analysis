#Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import yfinance as yf

#Load data, add a new field
apple = yf.Ticker("AAPL")
hist = apple.history(start="2000-01-01", end="2020-09-30", actions=False)
percent = (hist["Close"]-hist["Open"])/hist["Open"]*100
hist["%Increase"] = percent

#Explore dataset
hist.head()
hist.dtypes

#Create train set and test set
train_set_size = 5000
data = np.array(hist)
train_data = data[:train_set_size]
test_data = data[train_set_size:]
x_train = data[:train_set_size,:-1]
y_train = data[:train_set_size,-1]
x_test = data[train_set_size:,:-1]
y_test = data[train_set_size:,-1]
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)
#Create a sequence
train_window = 5 #a 5-day window for stock prices
def create_inout_sequences(input_data, train_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-train_window):
        train_seq = input_data[i:i+train_window]
        train_label = input_data[i+train_window:i+train_window+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
