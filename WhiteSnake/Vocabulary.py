from torch.utils import data
from sklearn import preprocessing
import numpy as np
import pickle
import csv
import os
import functools

class Dataset(data.Dataset):
    def __init__(self, ratio=0.6, windowSize=3):
        currentDir = os.path.dirname(os.path.abspath(__file__))
        data = list(csv.reader(open('{}/white-snake-preprocessor.csv'.format(currentDir), 'r')))
        data = sum(data[:int(len(data)*ratio)], [])
        data = list(''.join(data))
        data = list(filter(lambda x: x != 'n', data))
        data = [x + 'n' if x == '\\' else x for x in data]

        self.windowSize = windowSize
        self.labelEncoder = pickle.load(open('{}/label.pickle'.format(currentDir), 'rb'))
        self.lengths = list(map(lambda x: len(x), data))
        self.data = data

    def __getitem__(self, index):
        window = [self.data[index + i] for i in range(self.windowSize)]
        labels = self.labelEncoder.transform(window)
        return tuple(labels)

    def __len__(self):
        return len(self.data) - (self.windowSize - 1)