"""
For this problem we will be using the LeNet-5 architecture since we will resize the image to (32 x 32) 

LeNet-5 architecture
Conv1 : in_chanels = 1, out_channels = 6
Conv2 : in_channels = 6, out_channels = 16
FC layer1 :  120 units
FC layer2 : 84 units

"""

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

class CNNModule(nn.Module):

    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 1, padding = 2),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.fc1 = nn.Linear(256*3*3, 1024)
        # self.drop = nn.Dropout(p=0.25)
        # self.fc2 = nn.Linear(120, 84)
        # self.grapheme = nn.Linear(84, 168)
        # self.vowel = nn.Linear(84, 11)
        # self.consonant = nn.Linear(84, 7)
        self.batch = nn.BatchNorm1d(1024)
        self.grapheme = nn.Linear(1024, 168)
        self.vowel = nn.Linear(1024, 11)
        self.consonant = nn.Linear(1024, 7)


    def forward(self, x):
        convlayer1 = self.conv1(x)
        convlayer2 = self.conv2(convlayer1)
        convlayer3 = self.conv3(convlayer2)
        preprocessed = convlayer3.view(convlayer3.size(0), -1)
        fclayer1 = self.fc1(preprocessed)
        fclayer1 = self.batch(fclayer1)
        # fclayer1 = self.drop(fclayer1)
        # fclayer2 = self.fc2(fclayer1)
        # graphemeout = self.grapheme(fclayer2)
        # vowelout = self.vowel(fclayer2)
        # consonantout = self.consonant(fclayer2)
        graphemeout = self.grapheme(fclayer1)
        vowelout = self.vowel(fclayer1)
        consonantout = self.consonant(fclayer1)
        return graphemeout, vowelout, consonantout

class LossFunc:
    def __init__(self, model):
        self.model = model

    def loss_func(self, lr=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        return optimizer, loss




