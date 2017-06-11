import numpy as np
import os
import sys
import warnings
from matplotlib import pyplot as plt
import matplotlib


class PointSelector:
    def __init__(self, fig):
        self.point = np.zeros((2, 1))

        self.fig = fig
        self.cid = None

    def onclick(self, event):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.point = np.array([[event.xdata], [event.ydata]]).astype(np.int)

    def get_point(self):
        plt.figure(self.fig.number)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.waitforbuttonpress()
        return self.point


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")
    matplotlib.interactive(True)

    print("---------------------------")
    print("Start of the script")

    figure = plt.figure()
    plt.plot([0, 1000], [0, 1000])
    plt.show()

    point_selector = PointSelector(figure)
    point = point_selector.get_point()
    print ('Initial position selected. x=%d, y=%d' % (point[0, 0], point[1, 0]))

    print "\nClick to finish process..."
    plt.waitforbuttonpress()

    print("==========================")
