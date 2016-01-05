#!/usr/bin/env python
#
# Copyright 2013-2016 Richard Pausch
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

from numpy import *
import sys

data = loadtxt(sys.argv[1], dtype=str)

format = data.shape
data = data.flatten()

for i in xrange(data.size):
    data[i] = data[i].replace(",", " ")

data = data.astype(float)
data = data.reshape((format[0], format[1] / 3, 3))

dataAbs = sqrt(data[:,:,0]**2 + data[:,:,1]**2 + data[:,:,2]**2)

import matplotlib.pyplot as plt
plt.imshow(dataAbs, interpolation='nearest')
plt.colorbar()
plt.show()

