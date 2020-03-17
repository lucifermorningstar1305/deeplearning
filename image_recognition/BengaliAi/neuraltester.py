import os
import torch
from torch.autograd import Variable
import torch.utils.data as Data


class NeuralTester:
    def __init__(self, dataX, model, batch_size=1, size=32):
        self.dataX = torch.from_numpy(dataX).float()
        self.test_data = Data.DataLoader(self.dataX, batch_size=batch_size)
        self.model = model
        self.size= size

    def evaluateModel(self):
        ygram = []
        yvowel = []
        yconsonant = []
        self.model.eval()
        for x in self.test_data:
            b_x = Variable(x).float()
            b_x = b_x.view(-1,1,self.size,self.size)
            output = self.model(b_x)
            _gram = torch.max(output[0], 1)[1].data.squeeze()
            _vowel = torch.max(output[1], 1)[1].data.squeeze()
            _consonant = torch.max(output[2], 1)[1].data.squeeze()
            ygram.append(_gram.tolist())
            yvowel.append(_vowel.tolist())
            yconsonant.append(_consonant.tolist())
        return ygram, yvowel, yconsonant