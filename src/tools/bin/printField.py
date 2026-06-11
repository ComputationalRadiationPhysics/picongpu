#!/usr/bin/env python

# SPDX-FileCopyrightText: Richard Pausch
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
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

dataAbs = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2 + data[:, :, 2] ** 2)

plt.imshow(dataAbs, interpolation="nearest")
plt.colorbar()
plt.show()
