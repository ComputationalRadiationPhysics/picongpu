"""
This file is part of the PIConGPU.

Copyright 2023-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import config
import testsuite.Output.Log as log
import sys
from . import _checkData as cD


def run_testsuite(direction: str = None,
                  dataDirection: str = None,
                  paramDirection: str = None,
                  jsonDirection: str = None,
                  resultDirection: str = None,
                  cmakeDirection: str = None):
    """
    Main function of the test-suite, starts and runs the test-suite.

    Input:
    -------
    direction:  str, optional
                Main path of the simulation
                All other paths are determined from this.
                All others only have to be defined if they
                deviate from the standard path or direction is None.
                Default: None

    dataDirection: str, optional
                   Path of the folder in which the results
                   of the simulation were saved
                   Default: None

    paramDirection: str, optional
                    Path of the folder in which the parameter
                    files .params of the simulation were saved.
                    Default: None

    jsonDirection: str, optional
                   Path of the folder to the json files
                   if used.
                   Default: None

    resultDirection: str, optional
                     Path of the folder where the results
                     of the test-suite should be saved
                     Default: None

    cmakeDirection: str, optional
                        Path of the folder to the cmake files
                        if used.
                        Default: None

    Raise:
    -------
    InputError:  If all directories are None.

    """
    # determine all other directories if only "direction" is specified
    if (cD.checkExistVariables(variable="direction") or
            direction is not None):
        direction = cD.checkDirection(variable="direction",
                                      direction=direction)
        if (dataDirection is None and not
                cD.checkExistVariables(variable="dataDirection")):
            dataDirection = cD.checkDirection(variable="dataDirection",
                                              direction=direction +
                                              "simOutput/")
        if (paramDirection is None and not
                cD.checkExistVariables(variable="paramDirection")):
            paramDirection = cD.checkDirection(variable="paramDirection",
                                               direction=direction +
                                               "input/include/" +
                                               "picongpu/param/")
        if (cmakeDirection is None and not
                cD.checkExistVariables(variable="cmakeDirection")):
            cmakeDirection = cD.checkDirection(variable="cmakeDirection",
                                               direction=direction +
                                               "input/")

    try:
        # read the Data
        from .Reader import _manager as read

        json, param, data = read.mainsearch(dataDirection,
                                            paramDirection,
                                            jsonDirection,
                                            cmakeDirection)

        # now we can determine the theory and simulation
        parameter = {**json, **param, **data}
        theory = config.theory(**parameter)
        simData = config.simData(**parameter)

        # calculate the deviation
        from .Math import _manager as dv
        max_diff, perc, acc_range, result = dv._calculate(theory, simData)

        # get the result
        from .Output import _manager as output

        if resultDirection is None and direction is not None:
            resultDirection = direction

        output._output(theory=theory,
                       simulation=simData,
                       max_diff=max_diff,
                       perc=perc,
                       acc_range=acc_range,
                       result=result,
                       direction=resultDirection,
                       parameter=parameter)
        if result:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception:

        log.errorLog()
        sys.exit(42)
