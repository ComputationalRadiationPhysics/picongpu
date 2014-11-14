import numpy as _numpy
import StringIO as _StringIO


def readFieldSlices(File):
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
    if type(File) is file:
        theFile = File
    elif type(File) is str:
        theFile = file(File, 'r')
    else:
        # if neither trow arrow
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
            data[y,x,:] = _numpy.genfromtxt(_StringIO.StringIO(fieldValue[1:-1]), delimiter=",")
            
    return data



