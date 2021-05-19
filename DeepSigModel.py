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
import datetime
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

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

df=pd.read_csv("stocks_1980_2020.csv", header =0, index_col=0)
df.index = pd.to_datetime(df.index)
#df.set_index('Dates', inplace =True)
# Computes the data frame of returns
df_returns=df.pct_change()
my_stocks = ['AAPL' ]
start_date = datetime.date(1980, 12, 15)
df_returns = df_returns.loc[start_date : ]
a = []
b = []

for i in my_stocks:
    df_stock = df_returns[i].fillna(0)
    _X, _y = stock_to_train_test_cl(df_stock, k=33)
    a.append(_X)
    b.append(_y)
    
X = torch.tensor(np.stack(a), dtype = torch.float32)
y = torch.tensor(np.stack(b), dtype = torch.float32)
X_train, X_test = torch.split(X, 8000, dim = 1)
y_train, y_test = torch.split(y, 8000, dim=1)

#making our deepsig model
class DeepSigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth):
        super(DeepSigNet, self).__init__()
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(64, 8),
                                          kernel_size=1,
                                          include_original=True,
                                          include_time=True)
        self.signature1 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # +1 because self.augment1 is used to add time
        
        sig_channels1 = signatory.signature_channels(channels=in_channels + 9,
                                                     depth=sig_depth)
        self.augment2 = signatory.Augment(in_channels=sig_channels1,
                                          layer_sizes=(64, 8),
                                          kernel_size=1,
                                          include_original=False,
                                          include_time=False)
        self.signature2 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # 4 because that's the final layer size in self.augment2
        sig_channels2 = signatory.signature_channels(channels=9,
                                                     depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels2, out_dimension)
        self.leakyrelu = nn.LeakyReLU()

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
        v = self.leakyrelu(d)
        e = self.linear(v)
        # e is a three dimensional tensor of shape (batch,stream, out_dimension)
        return e
 
num_epochs = 500 #1000 epochs
learning_rate = 0.001 #0.001 lr

in_channels = 33 #number of features
sig_depth = 3  # depth for the signature transformation
out_dimension = 1 #number of output classes  

loss_test = 0   #MSEtest storing variable


# Training the NeuralSig model
deepsignet = DeepSigNet(in_channels, out_dimension, sig_depth) #our DeepSig class 
criterion = RMSELoss()    # root mean-squared error for regression
optimizer = torch.optim.Adam(deepsignet.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        outputs = deepsignet.forward(X_train) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
        loss = criterion(outputs, y_train
 
        loss.backward() #calculates the loss of the loss function
 
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:                           # :if you wouuld like to see the loss progression 
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            
output2 = deepsignet.forward(X_test)
loss_test = criterion(output2, y_test)         # MSEtest values   
y_pred = torch.flatten(output2)
y_actual = torch.flatten(y_test)
TP = 0
TN = 0
FP = 0
FN = 0
for j, k in zip(y_pred, y_actual) :
    if (j >= 0) & (k >= 0):
        TP +=1
    if (j < 0) & (k < 0): 
        TN +=1
    if (j >=0) & (k < 0):
        FP +=1
    if (j < 0) & (k >=0) :
        FN +=1

acc_test = (TP+TN)/(TP+TN+FP+FN)
prec_test[0] = TP/(TP+FP) 
prec_test[1] = TN/(TN+FN)
recall_test[0] =  TP/(TP+FN)
recall_test[1] = TN/(TN+FP)
f1_score[0] = 2*(prec_test[0]*recall_test[0]) / (prec_test[0] + recall_test[0])
f1_score[1] = 2*(prec_test[1]*recall_test[1]) / (prec_test[1] + recall_test[1])



