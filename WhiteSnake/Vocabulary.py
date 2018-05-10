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
        windows = self.data[-(self.windowSize - 1):] + self.data + self.data[:self.windowSize]
        windows = self.labelEncoder.transform(windows)
        index += self.windowSize - 1

        leftWords = windows[length + index - self.windowSize:length + index - 1]
        rightWords = windows[length + index:length + index + self.windowSize - 1]
        rightWords = rightWords[::-1] # reverse array

        center = windows[length + index - 1].item()
        mapContexts = list(map(lambda x: [x[1].item()] * (x[0][0] + 1), list(np.ndenumerate(leftWords)) + list(np.ndenumerate(rightWords))))
        contextWords = list(reduce(lambda x, y: x + y, mapContexts))

        return (center, contextWords)

    def __len__(self):
        return len(self.data)
