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
from __future__ import division
import numpy as np

__doc__ = "This is the 'smooth' module which provides several functions that\n\
provide methods to smooth data from simulation or experiments.\n\
It can be applied to 1D and 2D data sets."


def __info__():
    """
    This is the 'smooth' module which provides several functions that
    provide methods to smooth data from simulation or experiments.
    It can be applied to 1D and 2D data sets.

    If you are running this module as executable program from your
    shell, you will now have a look at all manuals of the functions
    provided by this module.

    To contine press 'q'.
    """


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

    Parameters:
    -----------
    number int
           number to which the next odd number is requested
    larger bool (optinal, default=True)
           select wheter nnext odd number should be larger (True)
           or smaler (False) than number

    Return:
    -------
    returns next odd number

    """
    if number % 2 == 1:
        # in case number is odd
        return number
    elif number % 2 == 0:
        # in case number is even
        if larger:
            return number + 1
        else:
            return number - 1
    else:
        error_msg = ("ERROR: number (= {}) neither odd " +
                     "nor even").format(number)
        raise Exception(error_msg)


def gaussWindow(N, sigma):
    """
    This function returns N discrete points of a Gauss function
    with a standard deviation of sigma (in units of discrete points).
    The return values are symetric and strech from 0 to N-1.

    ATTENTION: this gauss function is NOT normalized.

    Parameters:
    -----------
    N     - int
            number of sample and return points
    sigma - float
            standard deviation in units of descrete points

    Returns:
    --------
    returns N symetric samples of e^(-0.5*(x/sigma)^2)
    """
    # +/- range bins to calculate
    length = (N/float(sigma))
    # not normalized
    return np.exp(-0.5 * (np.linspace(-length, length, N))**2)


def smooth(x, sigma, window_len=11, fkt=gaussWindow):
    """
    A function that returns smoothed 1D-data from given data.

    Parameters:
    -----------
    x           - numpy.ndarray (1D)
                  original (noisy) data
    sigma       - float
                  standard deviation used by the window function 'fkt'
    window_len  - int (optinal)
                  number of bins used for the window function 'fkt'
                  default: 11 bins
    fkt         - function (optional)
                  window function used for smoothing
                  default: smooth.gaussWindow

    Returns:
    --------
    returns smoothed data with samle length as x

    """
    # check input:
    if type(x) != np.ndarray:
        error_msg = "ERROR: input needs to by a 1D numpy array. " + \
                    "Data type is {}".format(type(x))
        raise Exception(error_msg)

    if len(x.shape) != 1:
        # not a 1D array
        error_msg = "ERROR: input needs to by a 1D numpy array. " + \
                    "Data shape is {}".format(x.shape)
        raise Exception(error_msg)

    # extending the data at the beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]

    w = fkt(window_len, sigma)  # compute window values

    # smooth data by convolution with window function
    #   smoothed data with borders
    y = np.convolve(w/w.sum(), s, mode='valid')
    #   usually window_len is odd, and int-devision is used
    overlap = window_len//2

    return y[overlap:len(y)-overlap]  # smoothed data without added borders


def smooth2D(data, sigma_x=10, len_x=50, sigma_y=10, len_y=50,
             fkt=gaussWindow):
    """
    This function smoothes the noisy data of a 2D array.

    Parameters:
    -----------
    data       - numpy.ndaray (2D)
                 original (noisy) data - needs to be a 2D array
    sigma_x    - float (optinal)
                 standard deviation of the window function 'fkt' in x-direction
                 default: 10 bins
    len_x      - int (optional)
                 number of bins used for the window function 'fkt' in
                 x-direction
                 default: 50
    sigma_y    - float (optinal)
                 standard deviation of the window function 'fkt' in y-direction
                 default: 10 bins
    len_y      - int (optinal)
                 number of bins used for the window function 'fkt' in
                 y-direction
                 default: 50
    fkt        - function (optinal)
                 window function
                 default: smooth.gaussWindow

    Returns:
    --------
    smooth 2D-data with same dimensions as 'data'

    """
    # check input
    if type(data) != np.ndarray:
        error_msg = "ERROR: input needs to by a 2D numpy array. " + \
                    "Data type is {}".format(type(data))
        raise Exception(error_msg)

    # make a copy since python is handling arrays by reference
    data_cp = data.copy()

    if len(data.shape) != 2:
        # not a 2D array
        error_msg = "ERROR: input needs to by a 2D numpy array. " + \
                    "Data shape is {}".format(data.shape)
        raise Exception(error_msg)

    # make add window bins (maximum value included)
    len_x = makeOddNumber(len_x)
    len_y = makeOddNumber(len_y)

    # smooth x
    for i in range(len(data_cp)):
        data_cp[i] = smooth(data_cp[i], sigma_x, window_len=len_x,
                            fkt=gaussWindow)

    # smooth y
    for j in range(len(data_cp[0])):
        data_cp[:, j] = smooth(data_cp[:, j], sigma_y, window_len=len_y,
                               fkt=gaussWindow)

    # return smoothed copy
    return data_cp


if __name__ == "__main__":
    # call all function manuals
    help(__info__)
    help(makeOddNumber)
    help(gaussWindow)
    help(smooth)
    help(smooth2D)
