import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelNorm(nn.Module):
    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x
class CNNLSTM(nn.Module):
 def __init__(self,batch_size,in_channels,num_target_classes,hidden_dim=256,lstm_h_dim=256,layer_norm=False):
   super(CNNLSTM, self).__init__()
   self.in_channels=in_channels
   self.lstm_h_dim=lstm_h_dim
   self.batch_size=batch_size
   self.hidden_dim=512
   self.num_target_classes=num_target_classes
   if layer_norm:
      normLayer=nn.ChannelNorm
   else:
      normLayer=nn.BatchNorm1d
   ## ConvEncoder (Similar to CPC)
   self.c0=nn.Conv1d(self.in_channels,self.hidden_dim,10, stride=5,padding=3)
   self.batchNorm0 = normLayer(self.hidden_dim)
   self.c1=nn.Conv1d(self.hidden_dim,self.hidden_dim,8, stride=4,padding=2)
   self.batchNorm1 = normLayer(self.hidden_dim)
   self.c2=nn.Conv1d(self.hidden_dim,self.hidden_dim,4, stride=2,padding=1)
   self.batchNorm2 = normLayer(self.hidden_dim)
   self.c3=nn.Conv1d(self.hidden_dim,self.hidden_dim,4, stride=2,padding=1)
   self.batchNorm3 = normLayer(self.hidden_dim)
   self.c4=nn.Conv1d(self.hidden_dim,self.hidden_dim,4, stride=2,padding=1)
   self.batchNorm4 = normLayer(self.hidden_dim)
   self.Down_sampling_factor=160
   ####### 2-layers of BLSTM #####################
   self.LSTM= nn.LSTM(self.hidden_dim, self.lstm_h_dim,2, batch_first=True)
   ############# Linear Layer #######################
   self.Linear=nn.Linear(self.lstm_h_dim,self.num_target_classes)
   self.log_softmax=torch.nn.LogSoftmax(dim=-1)
 def forward(self,x):
   x = F.relu(self.batchNorm0(self.c0(x)))
   x = F.relu(self.batchNorm1(self.c1(x)))
   x = F.relu(self.batchNorm2(self.c2(x)))
   x = F.relu(self.batchNorm3(self.c3(x)))
   x = F.relu(self.batchNorm4(self.c4(x)))
   #print(x[1,90:100,:],x[2,90:100,:])
   x=x.permute(0,2,1)
   self.LSTM.flatten_parameters()
   x,_= self.LSTM(x)
   #print(x[0,1,:],x[1,1,:])
   embedding=x[:,-1,:]
   x=self.log_softmax(self.Linear(x[:,-1,:]))
   return x,embedding
