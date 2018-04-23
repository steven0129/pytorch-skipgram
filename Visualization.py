import visdom

class CustomVisdom(object):
    def __init__(self, name='default', **kwargs):
        self.vis = visdom.Visdom(env=name, **kwargs)
    
    def text(self, id, content):
        try:
            self.vis.text(content, win=id)
        except:
            pass