import pandas as pd
import numpy  as np
import os
import argparse
from cnn_architecture import CNNModule, LossFunc
from resizing import Resizer
from neuraltrainer import NeuralTraining
from neuraltester import NeuralTester
import torch
import natsort
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='define the input file name')
parser.add_argument('-m', '--modelno', help='define the model number')
parser.add_argument('-t', '--traintest', help='define whether to train or test')
parser.add_argument('-e', '--epochs', help='define the number of epochs')
args = parser.parse_args()
if not int(args.traintest):

    df = pd.read_csv(os.path.join(os.getcwd(),args.filename))
    labels = pd.read_csv('train.csv')
    df = pd.merge(df, labels, on='image_id')
    df.drop(['image_id','grapheme'], axis=1, inplace=True)
    cnn = CNNModule().cuda()
    if os.path.exists(os.path.join(os.getcwd(),'models/model{}.pkl'.format(int(args.modelno)))):
        # cnn = CNNModule().cuda()
        cnn.load_state_dict(torch.load(os.path.join(os.getcwd(),'models/model{}.pkl'.format(int(args.modelno)))))
    # model = cnn.load_state_dict(torch.load(os.path.join(os.getcwd(),'models/model{}.pkl'.format(int(args.modelno))))) if os.path.exists(os.path.join(os.getcwd(),'models/model{}.pkl'.format(int(args.modelno)))) else cnn
    optimizer, loss = LossFunc(cnn).loss_func()
    NeuralTraining(df.loc[:, '0':'783'].values / 255., df.loc[:, 'grapheme_root':'consonant_diacritic'].values, cnn, loss, optimizer, int(args.modelno)).trainModel(int(args.epochs) if args.epochs != None else 500)

else:

    df = pd.read_parquet(os.path.join(os.getcwd(),args.filename), engine='pyarrow')
    labels = pd.read_csv('test.csv')
    images = Resizer().image_resize(df.loc[:, '0':'32331'])
    images = images.values / 255.
    files = natsort.natsorted(glob(os.path.join(os.getcwd(),'models/*.pkl')))
    # print(files)
    # for f in files:
    cnn = torch.load(files[int(args.modelno)],  map_location=lambda storage, loc: storage)
    # cnn = torch.load(f,  map_location=lambda storage, loc: storage)
    grapheme, vowels, consonants = NeuralTester(images, cnn).evaluateModel()
    csv = pd.merge(df, labels, on='image_id')
    csv = csv[['row_id']]
    csv['target'] = 0
    target = []
    for c, g, v in zip(consonants, grapheme, vowels):
        target.append(c)
        target.append(g)
        target.append(v)

    csv['target'] = target
    csv.to_csv(os.path.join(os.getcwd(),'tests/test{}.csv'.format(int(args.modelno))), index=False)


   
