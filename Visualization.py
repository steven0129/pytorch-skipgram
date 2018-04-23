import visdom
import numpy as np

class CustomVisdom(object):
    def __init__(self, name='default', **kwargs):
        self.vis = visdom.Visdom(env=name, **kwargs)
        self.line = {}
    
    def text(self, id, content):
        try:
            self.vis.text(content, win=id)
        except:
            pass

    def drawLine(self, id, value):
        try:
            if id not in self.line: self.line[id] = []
            self.line[id].append(value)
            self.vis.line(Y=np.array(self.line[id]), win=id)
        except:
            pass