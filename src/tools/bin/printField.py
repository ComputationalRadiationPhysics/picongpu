#!/usr/bin/env python
#
# Copyright 2013-2021 Richard Pausch
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
import numpy as np
import matplotlib.pyplot as plt
import sys


data = np.loadtxt(sys.argv[1], dtype=str)

format = data.shape
data = data.flatten()

for i in np.range(data.size):
    data[i] = data[i].replace(",", " ")

data = data.astype(float)
data = data.reshape((format[0], format[1] / 3, 3))

dataAbs = np.sqrt(data[:, :, 0]**2 + data[:, :, 1]**2 + data[:, :, 2]**2)

plt.imshow(dataAbs, interpolation='nearest')
plt.colorbar()
plt.show()
