import csv

class CSV(object):
    def __init__(self, name):
        self.writer = csv.writer(open(name + '.csv', 'w'))
        self.writer.writerows([['epoch', 'loss']])

    def write(self, value):
        self.writer.writerows(value)
