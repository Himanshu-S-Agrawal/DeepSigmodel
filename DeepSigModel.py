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
my_stocks = ['AAPL' , 'ABT', 'BA', 'BAC', 'BMY', 'C', 'CMCSA', 'CVX', 'DIS', 'GE' ]
start_date = datetime.date(2010, 1, 1)
df_returns = df_returns.loc[start_date : ]
a = []
b = []

for i in my_stocks:
    df_stock = df_returns[i].fillna(0)
    _X, _y = stock_to_train_test_cl(df_stock, k=20)
    a.append(_X)
    b.append(_y)
print (df_returns.shape)    
X = torch.tensor(np.stack(a), dtype = torch.float32)
y = torch.tensor(np.stack(b), dtype = torch.float32)
print (X.shape)
X_train, X_test = torch.split(X, 2200, dim = 1)
y_train, y_test = torch.split(y, 2200, dim=1)
print (X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
#add leakyrelu


#making our deepsig model
class DeepSigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth):
        super(DeepSigNet, self).__init__()
        self.augment1 = signatory.Augment(in_channels=in_channels,
                                          layer_sizes=(64,8),
                                          kernel_size=1,
                                          include_original=True,
                                          include_time=True)
        self.signature1 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # +1 because self.augment1 is used to add time
        
        sig_channels1 = signatory.signature_channels(channels=in_channels + 9,
                                                     depth=sig_depth)
        self.lstm1 = nn.LSTM(input_size=sig_channels1, hidden_size=20,
                          num_layers=2, batch_first = True)
        self.signature2 = signatory.Signature(depth=sig_depth,
                                              stream=True)

        # 4 because that's the final layer size in self.augment2
        sig_channels2 = signatory.signature_channels(channels=20,
                                                     depth=sig_depth)
        self.lstm2 = nn.LSTM(input_size=sig_channels2, hidden_size=20,
                          num_layers=2, batch_first = True)
        self.signature3 = signatory.Signature(depth=sig_depth,
                                              stream=True)
        sig_channels3 = signatory.signature_channels(channels=20,
                                                     depth=sig_depth)
        self.lstm3 = nn.LSTM(input_size=sig_channels3, hidden_size=20,
                          num_layers=2, batch_first = True, dropout = 0.1)
        self.linear = nn.Linear(20, out_dimension)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        a = self.augment1(inp)
        # a in a three dimensional tensor of shape (batch, stream, in_channels + 1)
        b = self.signature1(a, basepoint=True)
        # b is a three dimensional tensor of shape (batch, stream, sig_channels1)
        #v = self.leakyrelu(b)
        bn = nn.BatchNorm1d(b.shape[-1])
        b = b.permute(0, 2, 1)
        b_n = bn(b)
        b_n = b_n.permute(0, 2, 1)
        h_0 = torch.zeros(2, b.size(0), 20) #hidden state
        c_0 = torch.zeros(2, b.size(0), 20) #internal state
        output1, (h_n, c_n) = self.lstm1(b_n, (h_0, c_0))
        # c is a three dimensional tensor of shape (batch, stream, 4)
        d = self.signature2(output1, basepoint=True)
        # d is a three dimensional tensor of shape (batch,stream, sig_channels2)
        bn1 = nn.BatchNorm1d(d.shape[-1])
        d = d.permute(0, 2, 1)
        b_n1 = bn1(d)
        b_n1 = b_n1.permute(0, 2, 1)
        h_01 = torch.zeros(2, d.size(0), 20) #hidden state
        c_01 = torch.zeros(2, d.size(0), 20) #internal state
        output2, (h_n1, c_n1) = self.lstm2(b_n1, (h_01, c_01))
        f = self.signature3(output2, basepoint = True)
        bn2 = nn.BatchNorm1d(f.shape[-1])
        f = f.permute(0, 2, 1)
        b_n2 = bn2(f)
        b_n2 = b_n2.permute(0, 2, 1)
        h_02 = torch.zeros(2, f.size(0), 20) #hidden state
        c_02 = torch.zeros(2, f.size(0), 20) #internal state
        output3, (h_n2, c_n2) = self.lstm3(b_n2, (h_02, c_02))
        w = self.leakyrelu(output3)
        e = self.linear(w)
        # e is a three dimensional tensor of shape (batch,stream, out_dimension)
        return e
 
num_epochs = 200 #1000 epochs
learning_rate = 0.0005 #0.001 lr

in_channels = 20 #number of features
sig_depth = 3  # depth for the signature transformation
out_dimension = 1 #number of output classes  

loss_test = []  #MSEtest storing variable
acc_test =  []  #accuracy of direction
prec_test = [0, 0]
recall_test = [0, 0]
f1_score = [0, 0]
loss_train = []
accuracy = []
precision1 = []
precision2 = []
recall1 = []
recall2 = []
f1_score1 = []
f1_score2 = []


# Training the NeuralSig model
deepsignet = DeepSigNet(in_channels, out_dimension, sig_depth) #our DeepSig class
criterion = RMSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(deepsignet.parameters(), lr=learning_rate, weight_decay = 0.0001)
    
for epoch in range(num_epochs):
    outputs = deepsignet.forward(X_train) #forward pass
    #print (outputs.shape)
    optimizer.zero_grad(set_to_none=True) #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
    loss = criterion(outputs, y_train.reshape(outputs.shape))
 
    loss.backward() #calculates the loss of the loss function
 
    optimizer.step() #improve from loss, i.e backprop
#
    y_pred = torch.flatten(outputs)
    y_actual = torch.flatten(y_train)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j, k in zip(y_pred, y_actual) :
        if (j >= 0) & (k >= 0):
            tp +=1
        if (j < 0) & (k < 0): 
            tn +=1
        if (j >=0) & (k < 0):
            fp +=1
        if (j < 0) & (k >=0) :
            fn +=1
    recall_train = [0,0]
    f1_score = [0, 0]
    acc_train = (tp+tn)/(tp+tn+fp+fn)
    prec_train = [0, 0] 
    if (tp !=0) or (fp!=0):
        prec_train[0] = tp/(tp+fp) 
    if (tn!=0) or (fn!=0):
        prec_train[1] = tn/(tn+fn)
    recall_train[0] =  tp/(tp+fn)
    recall_train[1] = tn/(tn+fp)
    if (recall_train[0] != 0) or (prec_train[0] !=0):    
        f1_score[0] = 2*(prec_train[0]*recall_train[0]) / (prec_train[0] + recall_train[0])
    if (recall_train[1] !=0) or (prec_train[1] != 0):    
        f1_score[1] = 2*(prec_train[1]*recall_train[1]) / (prec_train[1] + recall_train[1])

    l1 = loss.item()
    loss_train.append(l1)
    accuracy.append(acc_train)
    precision1.append(prec_train[0])
    precision2.append(prec_train[1])
    recall1.append(recall_train[0])
    recall2.append(recall_train[1])
    f1_score1.append(f1_score[0])
    f1_score2.append(f1_score[1])

    if epoch % 10 == 0:   
                                # :if you wouuld like to see the loss progression 
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        print("Epoch: %d, accuracy: %1.5f" % (epoch, acc_train))
        print("precision: {}".format(prec_train))
        print("recall: {}".format(recall_train))
        print("f1_score: {}".format(f1_score))


output2 = deepsignet.forward(X_test)
loss_test = criterion(output2, y_test.reshape(output2.shape))         # MSEtest values   
y_pred = torch.flatten(output2)
y_actual = torch.flatten(y_test.reshape(output2.shape))
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
df_res = pd.DataFrame(np.column_stack([loss_train, accuracy, precision1, precision2, recall1, recall2, f1_score1, f1_score2]), columns = ['loss', 'accuracy', 'prec+', 'prec-', 'rec+', 'rec-', 'f1+', 'f1-'])
df_res.to_csv('res_deepsignet')
