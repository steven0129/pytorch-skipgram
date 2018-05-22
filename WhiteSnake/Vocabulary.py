from torch.utils import data
from sklearn import preprocessing
from functools import reduce
import numpy as np
import pickle
import csv
import os

class Dataset(data.Dataset):
    def __init__(self, ratio=0.6, windowSize=2):
        currentDir = os.path.dirname(os.path.abspath(__file__))
        data = list(
            csv.reader(
                open(
                    '{}/white-snake-preprocessor.csv'.format(currentDir),
                    'r')))
        data = sum(data[:int(len(data)*ratio)], [])
        data = list(''.join(data))
        data = list(filter(lambda x: x != 'n', data))
        data = [x + 'n' if x == '\\' else x for x in data]

        encoder = preprocessing.LabelEncoder()
        encoder.fit(data)

        self.windowSize = windowSize
        self.labelEncoder = encoder
        self.lengths = list(map(lambda x: len(x), data))
        self.data = data

    def __getitem__(self, index):
        length = self.lengths[index]
        return (index, self.windowSize, self.labelEncoder, self.lengths[index], self.data)

    def __len__(self):
        return len(self.data)
