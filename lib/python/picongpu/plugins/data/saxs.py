"""
This file is part of the PIConGPU.
Copyright 2017-2019 PIConGPU contributors
Authors: Juncheng E
"""

import numpy as np

# Read SAXS output from SAXS plugin into numpy array.
class SaxsData:
    def __init__(self, filename):
        self.filename = filename
        self.nq = self.getNq(filename)
        self.qx = np.zeros(self.nq)
        self.qy = np.zeros(self.nq)
        self.qz = np.zeros(self.nq)
        self.intensity = np.zeros(self.nq)
        with open(filename) as fin:
            # skip 2 lines
            for i in range(0,2):
                fin.readline()
            for i, line in enumerate(fin):
                sline = line.strip().split()
                self.qx[i] = float(sline[0])
                self.qy[i] = float(sline[1])
                self.qz[i] = float(sline[2])
                self.intensity[i] = float(sline[3])
    # get the number of q
    def getNq(self,filename):
        with open(filename) as fin:
             nq = int(fin.readline())
        fin.close()
        return nq
