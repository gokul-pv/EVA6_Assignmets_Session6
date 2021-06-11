from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    '''Defining the model
      set argument norm = b for Batch normalisation 
                    norm = l for Layer normalisation 
                    norm = g for Group normalisation 
    '''

    def __init__(self, norm = 'b'):
        super(Net, self).__init__()
        self.norm = norm

        if (self.norm == 'b'):
          self.conv1 = nn.Sequential(
              nn.Conv2d(1, 8, 3, padding=0, bias=False),       #Input: 28*28*1  Output: 26 * 26 * 8     RF = 3
              nn.ReLU(),
              nn.BatchNorm2d(8),
              nn.Dropout2d(0.02),

              nn.Conv2d(8, 10, 3, padding=0, bias=False),      #Input: 26*26*8  Output: 24 * 24 * 10    RF = 5 
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Dropout2d(0.02),

              nn.MaxPool2d(2, 2)                               #Input: 24*24*10  Output: 12 * 12 * 10   RF = 6       
          )
          
          self.conv2 = nn.Sequential(
              nn.Conv2d(10, 12, 3, padding=0, bias=False),     #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
              nn.ReLU(),
              nn.BatchNorm2d(12),
              nn.Dropout2d(0.02),

              nn.Conv2d(12, 14, 3, padding=0, bias=False),     #Input: 10*10*12    Output: 8 * 8 * 14   RF = 14
              nn.ReLU(),
              nn.BatchNorm2d(14),
              nn.Dropout2d(0.02),

              nn.Conv2d(14, 16, 3, padding=1, bias=False),     #Input: 8*8*16      Output: 8 * 8 * 16   RF = 18
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout2d(0.02),

              nn.Conv2d(16, 16, 3, padding=0, bias=False),     #Input: 6*6*16      Output: 6 * 6 * 16   RF = 22
              nn.ReLU(),   
              nn.BatchNorm2d(16),
              nn.Dropout2d(0.02)
          )

        if (self.norm == 'l'):
          self.conv1 = nn.Sequential(
              nn.Conv2d(1, 8, 3, padding=0, bias=False),       #Input: 28*28*1  Output: 26 * 26 * 8     RF = 3
              nn.ReLU(),
              nn.LayerNorm([8,26,26]),
              nn.Dropout2d(0.02),

              nn.Conv2d(8, 10, 3, padding=0, bias=False),      #Input: 26*26*8  Output: 24 * 24 * 10    RF = 5 
              nn.ReLU(),
              nn.LayerNorm([10,24,24]),
              nn.Dropout2d(0.02),

              nn.MaxPool2d(2, 2)                               #Input: 24*24*10  Output: 12 * 12 * 10   RF = 6       
          )
          
          self.conv2 = nn.Sequential(
              nn.Conv2d(10, 12, 3, padding=0, bias=False),     #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
              nn.ReLU(),
              nn.LayerNorm([12,10,10]),
              nn.Dropout2d(0.02),

              nn.Conv2d(12, 14, 3, padding=0, bias=False),     #Input: 10*10*12    Output: 8 * 8 * 14   RF = 14
              nn.ReLU(),
              nn.LayerNorm([14,8,8]),
              nn.Dropout2d(0.02),

              nn.Conv2d(14, 16, 3, padding=1, bias=False),     #Input: 8*8*16      Output: 8 * 8 * 16   RF = 18
              nn.ReLU(),
              nn.LayerNorm([16,8,8]),
              nn.Dropout2d(0.02),

              nn.Conv2d(16, 16, 3, padding=0, bias=False),     #Input: 6*6*16      Output: 6 * 6 * 16   RF = 22
              nn.ReLU(),   
              nn.LayerNorm([16,6,6]),
              nn.Dropout2d(0.02)
          ) 

        if (self.norm == 'g'):
          self.conv1 = nn.Sequential(
              nn.Conv2d(1, 8, 3, padding=0, bias=False),       #Input: 28*28*1  Output: 26 * 26 * 8     RF = 3
              nn.ReLU(),
              nn.GroupNorm(2,8),                               # 8 channel into two group
              nn.Dropout2d(0.02),

              nn.Conv2d(8, 10, 3, padding=0, bias=False),      #Input: 26*26*8  Output: 24 * 24 * 10    RF = 5 
              nn.ReLU(),
              nn.GroupNorm(2,10),                              # 10 channel into two group
              nn.Dropout2d(0.02),

              nn.MaxPool2d(2, 2)                               #Input: 24*24*10  Output: 12 * 12 * 10   RF = 6       
          )
          
          self.conv2 = nn.Sequential(
              nn.Conv2d(10, 12, 3, padding=0, bias=False),     #Input: 12*12*10  Output: 10 * 10 * 12   RF = 10 
              nn.ReLU(),
              nn.GroupNorm(2,12),                              # 12 channel into two group
              nn.Dropout2d(0.02),

              nn.Conv2d(12, 14, 3, padding=0, bias=False),     #Input: 10*10*12    Output: 8 * 8 * 14   RF = 14
              nn.ReLU(),
              nn.GroupNorm(2,14),                              # 14 channel into two group
              nn.Dropout2d(0.02),

              nn.Conv2d(14, 16, 3, padding=1, bias=False),     #Input: 8*8*16      Output: 8 * 8 * 16   RF = 18
              nn.ReLU(),
              nn.GroupNorm(2,16),                              # 16 channel into two group
              nn.Dropout2d(0.02),

              nn.Conv2d(16, 16, 3, padding=0, bias=False),     #Input: 6*6*16      Output: 6 * 6 * 16   RF = 22
              nn.ReLU(),   
              nn.GroupNorm(2,16),                              # 16 channel into two group
              nn.Dropout2d(0.02)
          )

        self.avgpool = nn.AvgPool2d(6)                             # Global avergaing pooling is done to convert 2D data to 1D data Output: 1 * 1 * 16
        self.conv3   = nn.Conv2d(16, 10, 1, padding=0, bias=False) # a 1x1 convolution on 1D data is same as fully connected layer having 16 inputs and 10 outputs

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
