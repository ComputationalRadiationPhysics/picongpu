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

import numpy
import sys

def makeOddNumber(number, larger=True):
    """
    This function takes a number and returns the next odd number.
    By default, the next larger number will be returned, but by
    setting larger=False the next smaller odd number will be 
    returned.

    Example:
    makeOddNumber(13) --> 13
    makeOddNumber(6) --> 7
    makeOddNumber(22, larger=False) --> 21

    """
    if number % 2 is 1:
        return number
    elif number % 2 is 0:
        if larger:
            return number +1
        else:
            return number -1
    else:
        print >> sys.stderr, "ERROR:", self.func_name, "number neather odd nor even"
        raise


def gaussWindow(N, sigma):
    """
    This function returns a Gauss function for N discrete points 
    and a standard deviation of sigma discrete points.

    ATTENTION: this gauss function is NOT normalized.

    """
    length = (N/float(sigma)) # +/- range bins  to  calculate
    return numpy.exp(-0.5 * (numpy.linspace(-length, length, N))**2) # not normalized


def smooth(x, sigma, window_len = 11, fkt=gaussWindow):
    """ 
    A function that returnes smoothed 1D-data from data x.

    x           - original (noisy) data
    sigma       - standard deviation used by the window function (fkt)
    window_len  - number of bins used for the window function (fkt)
                  default: 11 bins
    fkt         - window function
                  default: smooth.gaussWindow

    """
    # extending the data at the beginning and at the end
    # to apply the window at the borders
    s = numpy.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = fkt(window_len, sigma) # window values
    y = numpy.convolve(w/w.sum(), s, mode='valid') #smoothed data with borders
    overlap = window_len/2 # usually window_len is odd, and int-devision is used
    return y[overlap:len(y)-overlap] # smoothed data without added borders



def smooth2D(data, sigma_x = 10, len_x = 50, sigma_y = 10, len_y = 50, fkt=gaussWindow):
    """
    This function smoothes the noisy data of a 2D array.

    data       - original (noisy) data  
                 needs to be a 2D array
    sigma_x    - standard deviation of the window function (fkt) in x-direction
                 default: 10 bins
    len_x      - number of bins used for the window function (fkt) in x-direction
                 default: 50
    sigma_y    - standard deviation of the window function (fkt) in y-direction
                 default: 10 bins
    len_y      - number of bins used for the window function (fkt) in y-direction
                 default: 50
    fkt        - window function
                 default: smooth.gaussWindow

    """
    data_cp = data.copy() # make a copy since python is handling arrays by reference
    try:
        if len(data.shape) != 2:
            # not a 2D array
            raise
    
        # make add window bins (maximum value included)
        len_x = makeOddNumber(len_x)
        len_y = makeOddNumber(len_y)

        # smooth x
        for i in range(len(data_cp)):
            data_cp[i] = smooth(data_cp[i], sigma_x, window_len=len_x, fkt=gaussWindow)

        # smooth y
        for j in range(len(data_cp[0])):
            data_cp[:, j] = smooth(data_cp[:, j], sigma_y, window_len=len_y, fkt=gaussWindow)

        # return smoothed copy
        return data_cp
    except:
        print >> sys.stderr, "ERROR:", self.func_name, "input needs to by a 2D numpy array"
        raise



