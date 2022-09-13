"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import sys
from time import localtime, strftime
import argparse
import os

lt = localtime()
date = strftime("date: %d.%m.%Y", lt)
timeOfDay = strftime("time: %H:%M:%S", lt)

parser = argparse.ArgumentParser(description="Starts the test suite"
                                 " for the KHI growth rate")

parser.add_argument("paramDir", help="direction/to/.param/files",
                    type=str)

parser.add_argument("simDir", help="direction/to/simOutput",
                    type=str)

args = parser.parse_args()

try:
    sys.path.append(os.getcwd())
    from TestKHI import TestKHI
    from ViewerKHI import ViewerKHI

    field = ["total", "B_x", "B_y", "B_z", "E_x", "E_y", "E_z"]

    # write the testresult
    y = TestKHI(args.paramDir, args.simDir)
    result = y.createResultLog(args.simDir)

    plotter = ViewerKHI(args.paramDir, args.simDir)
    plotter.plotGrowthRate(field, True, True, True)
    sys.exit(result)

except Exception:
    error0 = str(sys.exc_info()[0])
    error1 = str(sys.exc_info()[1])
    error2 = str(sys.exc_info()[2])

    # print error0 + error1 + error2
    fobj_out = open(args.simDir + "/khi_error.log", "w")
    fobj_out.write(date + " " + timeOfDay + "\n")
    fobj_out.write("\n")
    fobj_out.write(error0 + " " + error1 + " " + error2 + "\n")
    fobj_out.close()

    sys.exit(42)
