import numpy as _numpy
import StringIO as _StringIO


def readFieldSlices(File):
    # case: file or filename
    if type(File) is file:
        theFile = File
    elif type(File) is str:
        theFile = file(File, 'r')
    else:
        raise IOError("the argument - {} - is not a file".format(File))

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

    data = _numpy.empty((N_y, N_x, 3))

    theFile.seek(0)

    # go through all valid lines
    for y in range(N_y):
        line = theFile.readline().split()
        # go through all valid field vectors
        for x in range(N_x):
            fieldValue = line[x]
            data[y,x,:] = _numpy.genfromtxt(_StringIO.StringIO(fieldValue[1:-1]), delimiter=",")
            
    return data
