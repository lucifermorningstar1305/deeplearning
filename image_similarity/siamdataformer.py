import numpy as np
import pandas as pd
import time 
from tensorflow.keras.datasets import mnist
class Dataset():
    images_train = np.array([])
    image_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def _get_siamese_similar_pair(self):
        label = np.random.choice(self.unique_train_label)
        l, r = np.random.choice(self.map_train_label_indices[label], 2, replace = False)
        return l, r, 1

    def _get_siamese_dissimilar_pair(self):
        label_l, label_r = np.random.choice(self.unique_train_label,2, replace= False)
        l = np.random.choice(self.map_train_label_indices[label_l])
        r = np.random.choice(self.map_train_label_indices[label_r])
        return l, r, 0

    def _get_siamese_pair(self):
        if np.random.random() < 0.5:
            return self._get_siamese_similar_pair()
        else:
            return self._get_siamese_dissimilar_pair()
        
    def get_siamese_batch(self, n):
        idx_left, idx_right, labels = [], [], []
        for _ in range(n):
            l, r, x = self._get_siamese_pair()
            idx_left.append(l)
            idx_right.append(r)
            labels.append(x)

        return self.Xtrain[idx_left, :], self.Xtrain[idx_right, :], np.expand_dims(labels, axis=1)


class DatasetFormer(Dataset):
    def __init__(self):

        (self.Xtrain, self.ytrain), (self.Xtest, self.ytest) = mnist.load_data() 
        self.Xtrain = np.expand_dims(self.Xtrain, axis=3) / 255.
        self.Xtest = np.expand_dims(self.Xtest, axis=3) / 255.
        self.ytrain = np.expand_dims(self.ytrain,axis=1)
        self.unique_train_label = np.unique(self.ytrain)
        self.map_train_label_indices = {label:np.flatnonzero(self.ytrain == label) for label in self.unique_train_label}
        