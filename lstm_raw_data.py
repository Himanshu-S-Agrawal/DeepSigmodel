# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:52:50 2021

@author: Himanshu Agrawal
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit

df=pd.read_csv("stocks_1980_2020.csv", index_col=0)
# Computes the data frame of returns
df_returns=df.pct_change()

stock_name='AAPL'
df_stock=df_returns[stock_name].dropna()


def stock_to_train_test_cl(df,k=30):
    '''
    :param df: pandas Series
        Stock values over a period of time
    :param k: int
        Number of rolling days used to predict stock at t+1
    :return:
        X: training matrix
        y: target values
    '''
    
    X = np.zeros([df.shape[0] - k, k])
    y = np.zeros(df.shape[0]-k)
    for i in range(df.shape[0]-k):
        X[i, :] = df.iloc[i:i+k].values
        y[i] = df[i+k]
    return X, y

X, y = stock_to_train_test_cl(df_stock, k=33)
tscv = TimeSeriesSplit()  # FOr making time-series split for training and testing

#Making the lstm model
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
 
   
num_epochs = 100 #1000 epochs
learning_rate = 0.01 #0.001 lr

input_size = 33 #number of features
hidden_size = 10 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes  

loss_test = [0]*5   #initializing MSEtest storing variable
i =0 

#training the lstm model
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train_tensors_final = torch.reshape(torch.Tensor(X_train),   (torch.Tensor(X_train).shape[0], 1, torch.Tensor(X_train).shape[1]))
    X_test_tensors_final = torch.reshape(torch.Tensor(X_test),  (torch.Tensor(X_test).shape[0], 1, torch.Tensor(X_test).shape[1])) 
    y_train_tensors_final = torch.Tensor(y_train).view( torch.Tensor(y_train).shape[0], 1)
    y_test_tensors_final = torch.Tensor(y_test).view(torch.Tensor(y_test).shape[0], 1)
    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[0]) #our lstm class
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm1.forward(X_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
        loss = criterion(outputs, y_train_tensors_final)
 
        loss.backward() #calculates the loss of the loss function
 
        optimizer.step() #improve from loss, i.e backprop
  
  
        if epoch % 10 == 0:                           # :if you wouuld like to see the loss progression uncomment
            print("Epoch: %d, loss: %1.6f" % (epoch, loss.item()))
            

    output2 = lstm1.forward(X_test_tensors_final)
    loss_test[i] = criterion(output2, y_test_tensors_final)
    i +=1

