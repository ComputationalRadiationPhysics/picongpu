#!/usr/bin/env python
#
# Copyright 2013 Richard Pausch
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

"""Fourier Transformation of data"""


from numpy import *
#from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import optimize

import sys

lambda_central = 800.0e-9 / 2.0
pulse_len_should = 20.0e-15 / 2.35482

delta_omega_rel = 5.0
c = 299792458.0

f_central = c / lambda_central
omega_central = 2.0 * pi * f_central

# load data
tempdata = loadtxt(sys.argv[1], usecols=(2,3))

x_data=tempdata[:,0]
data=tempdata[:,1]
data *= data

delta_x = x_data[2] - x_data[1]
delta_t = ( delta_x ) / c
#print "dx = ", delta_x
#print "dt = ", delta_t

N = len(data)
#print("Number of data points: %d" % N)

#plt.figure(0, figsize=(18,8))  # definiert Grafikfenster

#plt.subplot(111, autoscale_on=True) # mittiger Plot
#plt.title("Fourier Coefficients")
#plt.xlabel(r"$\omega = 2\cdot\pi\cdot f$")
#plt.ylabel(r"$c_n$")

# interesting window in frequency
omega_min = omega_central / delta_omega_rel
omega_max = omega_central * delta_omega_rel
delta_omega = 2.0*pi/(N*delta_t)
n_min = floor(omega_min / delta_omega)
n_max = ceil(omega_max / delta_omega)

omega = arange(N)*delta_omega
fft_data = abs(fft(data))

#plt.plot(omega[n_min:n_max], fft_data[n_min:n_max]
#          , marker='o', linestyle='None', color='r', label="FFT")
     # zeichnet FFT-Koeffizienten fuer Funktion 1

#plt.show()


# fit data
class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value
            
def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)


# giving initial parameters
mu = Parameter(omega_central)
sigma = Parameter(1.0 / sqrt(2.0) / pulse_len_should * 2.0)
height = Parameter(fft_data[n_min:n_max].max())
#print "sigma_start =", sigma()

# define your function:
def f(x): return height() * exp(-((x-mu())/sigma())**2)
def f2(x,m,s,h): return h * exp(-((x-m)/s)**2)

# fit! (given that data is an array with the data to fit)
fitted = fit(f, [mu, sigma, height], fft_data[n_min:n_max], omega[n_min:n_max])

fitted2 = optimize.curve_fit(f2, omega[n_min:n_max], fft_data[n_min:n_max], [mu(), sigma(), height()])

sigma_omega = fitted[0][1]
#print "sigma_omega =", fitted[0][1]
sigma_freq = fitted[0][1]/2.0/pi
#print "sigma_freq =", sigma_freq, "[Hz]"

sigma_t = 1.0 / sqrt(2.0) / sigma_omega * 2.0
print "sigma_fft: ", sigma_t, "[s]"
print "fwhm_fft: ", sigma_t * 2.354820045, "[s]"
