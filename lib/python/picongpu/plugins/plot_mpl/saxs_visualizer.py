"""
This file is part of the PIConGPU.
Copyright 2017-2019 PIConGPU contributors
Authors: Juncheng E
"""

from picongpu.plugins.data import SaxsData 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Visualize SAXS output from SAXS plugin with matplotlib.
class Visualizer:
    def __init__(self, filename):
        self.data = SaxsData(filename)
    def plot_xy(self,slicePoint = 0,sliceWidth = 0.05,markerSize=5):
        points = (self.data.qz <  slicePoint+sliceWidth/2.0) | (self.data.qz >  slicePoint-sliceWidth/2.0)
        x = self.data.qx[points]
        y = self.data.qy[points]
        intensity = self.data.intensity[points]
        self.createPlot(x,y,intensity,markerSize)
    def plot_yz(self,slicePoint = 0,sliceWidth = 0.05,markerSize=5):
        points = (self.data.qx <  slicePoint+sliceWidth/2.0) | (self.data.qx >  slicePoint-sliceWidth/2.0)
        y = self.data.qy[points]
        z = self.data.qz[points]
        intensity = self.data.intensity[points]
        self.createPlot(y,z,intensity,markerSize)
    def plot_xz(self,slicePoint = 0,sliceWidth = 0.05,markerSize=5):
        points = (self.data.qy <  slicePoint+sliceWidth/2.0) | (self.data.qy >  slicePoint-sliceWidth/2.0)
        x = self.data.qx[points]
        z = self.data.qz[points]
        intensity = self.data.intensity[points]
        self.createPlot(x,z,intensity,markerSize)
    def createPlot(self,x,y,intensity,markerSize=5):
        fig, ax = plt.subplots(1,1)
        cax = ax.scatter(x, y, c=intensity,linewidths=0.0,marker='s',s=markerSize,norm=LogNorm(vmin=1e0))
        ax.axis('equal')
        ax.axis('off')
        fig.colorbar(cax)
        plt.show()