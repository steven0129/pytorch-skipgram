from torch.utils import data
from sklearn import preprocessing
import numpy as np
import pickle
import csv
import os

class Dataset(data.Dataset):
    def __init__(self, ratio=0.6):
        currentDir = os.path.dirname(os.path.abspath(__file__))
        data = sum(data[:int(len(data)*ratio)], [])

        self.labelEncoder = pickle.load(open('{}/label.pickle'.format(currentDir), 'rb'))
        self.length = list(map(lambda x: len(x), data))
        self.data = data

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass