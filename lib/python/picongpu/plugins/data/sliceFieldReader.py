# Copyright 2014-2021 Richard Pausch, Klaus Steiniger
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

import numpy as _numpy
from io import IOBase


def FieldSliceData(File):
    """
    Function to read one data file from PIConGPUs SliceFieldPrinter plug-in
    and returns the field data as array of size [N_y, N_x, 3], with
    N_y, N_x the size of the slice.

    Parameters:
    -----------
    File: either a file of type file which points to the data file or
          a filename of type str with the path to the data file

    Returns:
    --------
    numpy-array with field data
    """
    # case: file or filename
    if isinstance(File, IOBase):
        theFile = File
    elif type(File) is str:
        theFile = open(File, 'r')
    else:
        # if neither throw error
        raise IOError("the argument - {} - is not a file".format(File))

    # determine size of slice:
    N_x = None
    N_y = 0

    for line in theFile:
        # count number of vectors in line
        N_x_local = line.count('{')

        # check whether number of vectors stays constant
        if N_x is not None:
            # none avoid initial line with no entry number yet
            if N_x_local != N_x and N_x_local > 0:
                raise IOError("number of entries differs between lines")

        # add number of lines with valid entries
        if N_x_local > 0:
            N_y += 1
            # set N_x if line was valid
            N_x = N_x_local

    # create data storage for slice of field data
    data = _numpy.empty((N_y, N_x, 3))

    # read file again
    theFile.seek(0)

    # go through all valid lines
    for y in range(N_y):
        line = theFile.readline().split()
        # go through all valid field vectors
        for x in range(N_x):
            fieldValue = line[x]
            data[y, x, :] = [float(x) for x in fieldValue[1:-1].split(',')]
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    # set up command line argument parser
    parser = argparse.ArgumentParser(
        description='''This is just the test case for the python module to load
                       data from PIConGPUs SliceFieldPrinter plug-in into
                       python''',
        epilog="For further questions, ask Richard Pausch.")

    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help="File with data of the SliceFieldPrinter plugin.")

    args = parser.parse_args()

    # load data from file using this module
    data = FieldSliceData(args.file)

    # show data (field_x only)
    plt.imshow(data[:, :, 0])
    plt.colorbar()
    plt.show()
