#!/usr/bin/env python
#
# Copyright 2013-2017 Heiko Burau, Axel Huebl
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

#___________P A R A M E T E R S___________

omega_plasma = 6.718e13     # SI unit: 1/s
v_th = 1.0e8                # SI unit: m/s
c = 2.9979e8                # SI unit: m/s
delta_t = 2.5e-15           # SI unit: s
delta_z = c * delta_t       # SI unit: m

#_________________________________________

from numpy import *
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

data_trans = loadtxt("eField_zt_trans.dat")
data_long = loadtxt("eField_zt_long.dat")

N_z = len(data_trans[:,0])
N_t = len(data_trans[0,:])

omega_max = pi*(N_t-1)/(N_t*delta_t)/omega_plasma
k_max = pi * (N_z-1)/(N_z*delta_z)

# __________________transversal plot______________________

ax = plt.subplot(211, autoscale_on=False, xlim=(-k_max, k_max), ylim=(-1, 10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.2e'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

plt.xlabel(r"$k [1/m]$")
plt.ylabel(r"$\omega / \omega_{pe} $")

data_trans = fft.fftshift(fft.fft2(data_trans))

plt.imshow(abs(data_trans), extent=(-k_max, k_max, -omega_max, omega_max), aspect='auto', interpolation='nearest')
plt.colorbar()

# plot analytical dispersion relation
x = linspace(-k_max, k_max, 200)
y = sqrt(c**2 * x**2 + omega_plasma**2)/omega_plasma
plt.plot(x, y, 'r--', linewidth=1)

# ___________________longitudinal plot_____________________

ax = plt.subplot(212, autoscale_on=False, xlim=(-k_max, k_max), ylim=(-1, 10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%2.2e'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

plt.xlabel(r"$k [1/m]$")
plt.ylabel(r"$\omega / \omega_{pe} $")

data_long = fft.fftshift(fft.fft2(data_long))

plt.imshow(abs(data_long), extent=(-k_max, k_max, -omega_max, omega_max), aspect='auto', interpolation='nearest')
plt.colorbar()

# plot analytical dispersion relation
x = linspace(-k_max, k_max, 200)
y = sqrt(3 * v_th**2 * x**2 + omega_plasma**2)/omega_plasma
plt.plot(x, y, 'r--', linewidth=1)

plt.show()
