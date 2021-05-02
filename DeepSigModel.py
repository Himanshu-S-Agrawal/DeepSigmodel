# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:57:17 2021

@author: Himanshu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import signatory
from sklearn.model_selection import TimeSeriesSplit

df=pd.read_csv("stocks_1980_2020.csv", index_col=0)
# Computes the data frame of returns
df_returns=df.pct_change()
#choose the stock to run the model on
stock_name='AAPL'
df_stock=df_returns[stock_name].dropna()

#Structuring the csv data for use in the model
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
tscv = TimeSeriesSplit()             # FOr making time-series split for training and testing
#making our deepsig model
class DeepSigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth):
        super(DeepSigNet, self).__init__()
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(),
                                          kernel_size=1,
                                          include_original=True,
                                          include_time=True)
        self.signature1 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # +1 because self.augment1 is used to add time
        
        sig_channels1 = signatory.signature_channels(channels=in_channels + 1,
                                                     depth=sig_depth)
        self.augment2 = signatory.Augment(in_channels=sig_channels1,
                                          layer_sizes=(8,4),
                                          kernel_size=1,
                                          include_original=False,
                                          include_time=False)
        self.signature2 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # 4 because that's the final layer size in self.augment2
        sig_channels2 = signatory.signature_channels(channels=4,
                                                     depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels2, out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        a = self.augment1(inp)
        if a.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # a in a three dimensional tensor of shape (batch, stream, in_channels + 1)
        b = self.signature1(a, basepoint=True)
        # b is a three dimensional tensor of shape (batch, stream, sig_channels1)
        c = self.augment2(b)
        if c.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # c is a three dimensional tensor of shape (batch, stream, 4)
        d = self.signature2(c, basepoint=True)
        # d is a three dimensional tensor of shape (batch,stream, sig_channels2)
        e = self.linear(d)
        # e is a three dimensional tensor of shape (batch,stream, out_dimension)
        return e
 
num_epochs = 500 #1000 epochs
learning_rate = 0.01 #0.001 lr

in_channels = 33 #number of features
sig_depth = 3  # depth for the signature transformation
out_dimension = 1 #number of output classes  

loss_test = [0]*5   #MSEtest storing variable
i = 0


# Training the NeuralSig model
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train_tensors_final = torch.reshape(torch.Tensor(X_train),   (1, torch.Tensor(X_train).shape[0],  torch.Tensor(X_train).shape[1]))
    X_test_tensors_final = torch.reshape(torch.Tensor(X_test),  (1, torch.Tensor(X_test).shape[0], torch.Tensor(X_test).shape[1])) 
    y_train_tensors_final = torch.Tensor(y_train).view(1, torch.Tensor(y_train).shape[0], 1)
    y_test_tensors_final = torch.Tensor(y_test).view(1, torch.Tensor(y_test).shape[0], 1)
    deepsignet = DeepSigNet(in_channels, out_dimension, sig_depth) #our DeepSig class
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(deepsignet.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        outputs = deepsignet.forward(X_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
        loss = criterion(outputs, y_train_tensors_final)
 
        loss.backward() #calculates the loss of the loss function
 
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:                           # :if you wouuld like to see the loss progression 
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            
    output2 = deepsignet.forward(X_test_tensors_final)
    loss_test[i] = criterion(output2, y_test_tensors_final)         # MSEtest values
    i +=1          



