import os
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import datetime
class NeuralTraining:
    def __init__(self, dataX, dataY, model, loss, optimizer, model_no, batch_size=256, size=28):
        self.dataX = torch.Tensor(dataX)
        self.dataY = torch.Tensor(dataY)
        train_dataset = Data.TensorDataset(self.dataX, self.dataY)
        self.train_data = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.model_no = model_no
        self.summary = SummaryWriter(comment='BengaliAI model no. : {}'.format(model_no))
        self.size=size

    def trainModel(self, epochs = 500):
        
        for i in tqdm(range(epochs)):
            running_loss = 0
            running_accuracy = 0
            self.model.train()
            quater = epochs // 4
            for x,y in self.train_data:
                b_x = Variable(x.cuda()).float()
                b_yg = Variable(y[:,0].cuda()).long()
                b_yv = Variable(y[:,1].cuda()).long()
                b_yc = Variable(y[:,2].cuda()).long()
                b_x = b_x.view(-1,1,self.size, self.size)
                output = self.model(b_x)
                # self.summary.add_graph(self.model, output)
                loss1 = self.loss(output[0], b_yg)
                loss2 = self.loss(output[1], b_yv)
                loss3 = self.loss(output[2], b_yc)
                loss_combined = loss1 + loss2 + loss3
                self.optimizer.zero_grad()
                loss_combined.backward()
                self.optimizer.step()
                acc1 = (torch.max(output[0],1)[1].data.squeeze() == b_yg).sum().item()/float(b_yg.size(0))
                acc2 = (torch.max(output[1],1)[1].data.squeeze() == b_yv).sum().item()/float(b_yv.size(0))
                acc3 = (torch.max(output[2],1)[1].data.squeeze() == b_yc).sum().item()/float(b_yc.size(0))
                acc = (acc1 + acc2 + acc3) / 3.
                running_loss += loss_combined * 256
                running_accuracy += acc * 256

            running_loss /= self.dataX.size(0)
            running_accuracy /= self.dataX.size(0)
            self.summary.add_scalar("Loss", running_loss, i)
            self.summary.add_scalar("Accuracy", running_accuracy,i)
            # self.summary.add_histogram("Input",self.dataX.data.cpu().numpy(),i)
            self.summary.add_histogram("Conv Layer-1",self.model.conv1[0].weight.data.cpu().numpy(),i)
            self.summary.add_histogram("Conv Layer-2",self.model.conv2[0].weight.data.cpu().numpy(),i)
            self.summary.add_histogram("Conv Layer-3",self.model.conv3[0].weight.data.cpu().numpy(),i)
            self.summary.add_histogram("FC-Layer-1",self.model.fc1.weight.data.cpu().numpy(), i)
            # self.summary.add_histogram("FC-Layer-2",self.model.fc2.weight.data.cpu().numpy(), i)
            torch.save(self.model.state_dict(), os.path.join(os.getcwd(),'models/model{}.pkl'.format(self.model_no)))



        self.summary.close()

    

