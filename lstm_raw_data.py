# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:52:50 2021

@author: Himanshu Agrawal
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
my_stocks = ['AAPL', 'ABT', 'BA', 'BAC', 'BMY', 'C', 'CMCSA', 'CVX', 'DIS', 'GE' ]
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
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 
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

loss_test = 0   #initializing MSEtest storing variable


#training the lstm model
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[0]) #our lstm class
criterion = RMSELoss()    # root mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
    loss = criterion(outputs, y_train)
 
    loss.backward() #calculates the loss of the loss function
 
    optimizer.step() #improve from loss, i.e backprop
  
  
    if epoch % 10 == 0:                           # :if you wouuld like to see the loss progression uncomment
        print("Epoch: %d, loss: %1.6f" % (epoch, loss.item()))
            

output2 = lstm1.forward(X_test)
loss_test = criterion(output2, y_test)         # RMSEtest values   
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
