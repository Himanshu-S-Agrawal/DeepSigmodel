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
my_stocks = ['AAPL' , 'ABT', 'BA', 'BAC', 'BMY', 'C', 'CMCSA', 'CVX', 'DIS', 'GE']
start_date = datetime.date(2000, 1, 1)
df_returns = df_returns.loc[start_date : ]
a = []
b = []

for i in my_stocks:
    df_stock = df_returns[i].fillna(0)
    _X, _y = stock_to_train_test_cl(df_stock, k=33)
    a.append(_X)
    b.append(_y)
print (df_returns.shape)    
X = torch.tensor(np.stack(a), dtype = torch.float32)
y = torch.tensor(np.stack(b), dtype = torch.float32)
print (X.shape)
X_train, X_test = torch.split(X, 4200, dim = 1)
y_train, y_test = torch.split(y, 4200, dim=1)
print (X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#add leakyrelu

# Making the NeuralSig model
class SigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth):
        super(SigNet, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(64, 8),
                                         kernel_size=1,
                                         activation = nn.LeakyReLU(),
                                         include_original=True,
                                         include_time=True)
        self.signature = signatory.Signature(depth=sig_depth, stream = True)
        # +1 because signatory.Augment is used to add time as well
        sig_channels = signatory.signature_channels(channels=in_channels + 9,
                                                    depth=sig_depth)
        self.linear = nn.Linear(sig_channels,
                                      out_dimension)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=True)
        # y is a three dimensional tensor of shape (batch, stream, terms), corresponding to
        # the terms of the signature
        v = self.leakyrelu(y)
        z = self.linear(v)
        # z is a Three dimensional tensor of shape (batch, stream, out_dimension)
        return z
 
num_epochs =  500   #change to 1000 if better fit possible 
learning_rate = 0.001 # or 0.01 lr

in_channels = 33 #number of features

sig_depth = 3  #depth of the signature for truncation

out_dimension = 1 #number of output classes  
loss_test = []  #MSEtest storing variable
acc_test =  []  #accuracy of direction
prec_test = [0, 0]
recall_test = [0, 0]
f1_score = [0, 0]

# Training the NeuralSig model

signet = SigNet(in_channels, out_dimension, sig_depth) #our NeuralSig class
#criterion = torch.nn.MSELoss()    # mean-squared error for regression
criterion = RMSELoss()
optimizer = torch.optim.Adam(signet.parameters(), lr=learning_rate)
    
for epoch in range(num_epochs):
    outputs = signet.forward(X_train) #forward pass
    print (outputs.shape)
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
    loss = criterion(outputs, y_train.reshape(outputs.shape))
 
    loss.backward() #calculates the loss of the loss function
 
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 1 == 0:                           # :if you wouuld like to see the loss progression 
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

output2 = signet.forward(X_test)
loss_test = criterion(output2, y_test.reshape(output2.shape))         # MSEtest values   
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

print (acc_test, f1_score, prec_test, prec_train)
