from torch.utils import data
from sklearn import preprocessing
import numpy as np
import pickle
import csv

class WhiteSnake(data.Dataset):
    def __init__(self, ratio=0.6, seed=1):
        self.labelEncoder = pickle.load(open('./data/label.pickle', 'rb'))
        data = list(csv.reader(open('./data/white-snake-preprocessor.csv', 'r')))
        data = sum(data[:int(len(data)*ratio)], [])
        self.length = list(map(lambda x: len(x), data))
        self.data = data

    def __getitem__(self, index):
        seqX = list(self.data[index])
        seqY = list(self.data[index + 1])
        seqX = np.pad(self.labelEncoder.transform(seqX), (0, max(self.length) - len(seqX)), 'constant').tolist()
        seqY = np.pad(self.labelEncoder.transform(seqY), (0, max(self.length) - len(seqY)), 'constant').tolist()
        return (seqX, seqY)

    def __len__(self):
        return len(self.data) - 1
