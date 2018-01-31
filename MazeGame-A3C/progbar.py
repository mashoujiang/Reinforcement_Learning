__author__ = 'shoujiang'
from blessings import Terminal
from progressive.bar import Bar
from progressive.tree import ProgressTree, Value, BarDescriptor

class ProgBar(object):
    
    def __init__(self,max_epoch):
        self.leaf_values = [Value(0) for i in range(1)]
        self.max_epoch = max_epoch
        self.epoch = 0
        self.test_d = {
                'Training Processing:': BarDescriptor(value=self.leaf_values[0],
                                                      type=Bar, 
                                                      kwargs=dict(max_value=self.max_epoch))
                }
        self.t = Terminal()
        self.n = ProgressTree(term=self.t)
        self.n.make_room(self.test_d)


    def incr_value(self):
        if self.epoch < self.max_epoch:
            self.leaf_values[0].value +=1
            self.epoch +=1

    def we_done(self):
        if self.epoch == self.max_epoch:
            return True
        else:
            return False

    def show_progbar(self):
        self.n.cursor.restore()
        self.incr_value()
        self.n.draw(self.test_d)

if __name__ == '__main__':
    pb = ProgBar(10)
    while not pb.we_done():
        pb.show_progbar()

