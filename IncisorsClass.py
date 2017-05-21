import numpy as np
import cv2
from matplotlib import pyplot as plt
import myLib
import procrustes_analysis
import os
import sys


class IncisorShape(procrustes_analysis.ObjectShape):
    'Doc string'

    def __init__(self, numImg, numTth):
        self.numImage = numImg
        self.numTooth = numTth
        
        self.pathLandmarks = "Project Data/_Data/Landmarks/original/landmarks" + str(numImg) + "-" + str(numTth) + ".txt"
        
        points = np.loadtxt(self.pathLandmarks)
        self.landmarks = np.zeros((2,40))
        self.landmarks[0,:] = points[::2]
        self.landmarks[1,:] = points[1::2]
        del points
        
        self.pathRadiograph = "Project Data/_Data/Radiographs/" + str(numImg).zfill(2) + ".tif"
        self.radiograph = np.zeros((1,1))
        self.radiograph = cv2.imread(self.pathRadiograph,0)
        
        self.pathSegmentation = "Project Data/_Data/Segmentations/" + str(numImg).zfill(2) + "-" + str(numTth-1) + ".png"        
        self.segmentation = np.zeros((1,1))
        self.segmentation = cv2.imread(self.pathSegmentation,0)
        
        
        
    def getNumImage(self):
        return self.numImage
        
    def showRadiograph(self, position): 
        plt.figure()
        plt.imshow(self.radiograph, cmap = 'gray', interpolation = 'bicubic')
        x = self.landmarks[0,:]
        y = self.landmarks[1,:]
        maxx = np.amax(x) + 10
        minx = np.amin(x) - 10 
        maxy = np.amax(y) + 10
        miny = np.amin(y) - 10
        plt.plot(x,y, 'r.',markersize=10)
        plt.plot(x[0],y[0], 'w.',markersize=10)
        plt.plot(x[1],y[1], 'b.',markersize=10)
        plt.plot(x[20],y[20], 'm.',markersize=10)
        plt.plot(x[21],y[21], 'b.',markersize=10)
        axes = plt.gca()
        axes.set_xlim([minx,maxx])
        axes.set_ylim([maxy,miny])
        myLib.move_figure('', np.hstack((position,np.array([2*(maxx-minx),maxy-miny]))))
        plt.show(block=False)
        return 2*(maxx-minx)
    
    def showSegmentation(self):
        plt.figure()
        plt.imshow(self.segmentation, cmap = 'gray', interpolation = 'bicubic')
        myLib.move_figure('bottom-right')
        x = self.landmarks[0,::5]
        y = self.landmarks[1,::5]
        plt.plot(x,y, 'r.',linewidth=1.0)    
        plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[0]))
    
    
    



