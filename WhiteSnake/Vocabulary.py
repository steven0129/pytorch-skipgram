from torch.utils import data
from sklearn import preprocessing
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

        self.windowSize = windowSize
        self.labelEncoder = pickle.load(
            open('{}/label.pickle'.format(currentDir), 'rb'))
        self.lengths = list(map(lambda x: len(x), data))
        self.data = data

    def __getitem__(self, index):
        length = self.lengths[index]
        windows = self.data[-(self.windowSize - 1):] + self.data + self.data[:self.windowSize]
        windows = self.labelEncoder.transform(windows).tolist()
        index += self.windowSize - 1

        contexts1 = windows[length + index - self.windowSize:length + index - 1]
        contexts2 = windows[length + index:length + index + self.windowSize - 1]
        center = windows[length + index - 1]
        contextWords = []

        for i, context in enumerate(contexts1):
            contextWords.extend([context] * (i + 1))

        contexts2.reverse()
        for i, context in enumerate(contexts2):
            contextWords.extend([context] * (i + 1))

        return tuple((center, contextWords))

    def __len__(self):
        return len(self.data)
