"""
This file is part of the PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Test suite log files

Routines in this module:

resultLog(direction:str = None, title:str = None)
errorLog()
"""

__all__ = ['resultLog', 'errorLog']

import sys
from time import localtime, strftime
import testsuite._checkData as cD


def resultLog(theory: float,
              value_sim: float,
              acceptance: float,
              perc_diff: int,
              result: bool,
              difference: float,
              direction: str = None,
              title: str = None,
              inputparameter=None):
    """
    Creates the file testresult.log, in which all essential
    parameters of the test and the result of the test are
    summarized.

    Input:
    -------
    theory :    float
                Value from the theory against which it was tested

    value_sim : float
                Value from the simulation against which it was tested

    acceptance : float
                 maximum deviation from the theoretical value in
                 percent

    perc_diff : int

    result :    bool
                result of the test, True if passed, otherwise False

    difference : float
                 difference between theory and value_sim

    direction : str, optional
                The directory in which the log file should be saved.
                If None or set in config.py, the value from config.py is
                used.

    title :     str, optional
                Description of the test case. If None or set in
                config.py, the value from config.py is used. If the title
                was not set at all, "no title" is used.

    inputparameter: dict, optional
                    all parameter that are used from the testsuite

    Raise:
    -------
    ValueError :
        If both values (direction and config.resultDirection) are None

    Example of testresult.log:
    -------

    date: 18.08.2022  time: 13:20:38

    Testcase: KHI Growthrate
    Theoretically expected value: 0.34662097116987617
    Value from simulation: 0.3473760369365579
    Acceptance range: (0.27729677693590093, 0.4159451654038514)
    Result of the test: passed
    Difference: -0.0007550657666817173
    Difference in percentage: -0.21736265211051872 %

    """

    try:

        lt = localtime()
        date = strftime("date: %d.%m.%Y", lt)
        timeOfDay = strftime("time: %H:%M:%S", lt)

        title = cD.checkVariables(variable="title",
                                  default="No title",
                                  parameter=title)

        direction = cD.checkDirection(variable="resultDirection",
                                      direction=direction)

        fobj_out = open(direction + "/testresult.log", "w+")
        fobj_out.write(date + " " + timeOfDay + "\n")
        fobj_out.write("\n")
        fobj_out.write("testcase: " + title + "\n")
        fobj_out.write("theoretically expected value: {}\n".format(theory))
        fobj_out.write("Value from simulation: {}\n".format(value_sim))
        fobj_out.write("acceptance: {}\n".format(acceptance))
        fobj_out.write("result of the test:{} \n".format(result))
        fobj_out.write("difference: {}\n".format(difference))
        fobj_out.write("difference in percentage: {}\n".format(perc_diff))
        for key in inputparameter.keys():
            fobj_out.write("input parameter: {}={}\n".format(
                key, inputparameter[key]))
        fobj_out.close()

    except Exception:
        errorLog()


def errorLog(direction: str = None):
    """
    Catches errors while executing the test-suite and saves
    them in the error.log file.

    Input:
    -------
    direction : str, optional
                The directory in which the error log file should be
                saved. If None or set in config.py, the value from
                config.py is used. If both are not set or the directory
                does not exist, the current working directory is used.
    """

    direction = cD.checkDirection(variable="resultDirection",
                                  direction=direction,
                                  errorhandling=True)

    lt = localtime()
    date = strftime("date: %d.%m.%Y", lt)
    timeOfDay = strftime("time: %H:%M:%S", lt)

    error0 = str(sys.exc_info()[0])
    error1 = str(sys.exc_info()[1])
    error2 = str(sys.exc_info()[2])

    # print error0 + error1 + error2
    fobj_out = open(direction + "/error.log", "w")
    fobj_out.write(date + " " + timeOfDay + "\n")
    fobj_out.write("\n")
    fobj_out.write(error0 + " " + error1 + " " + error2 + "\n")
    fobj_out.close()

    sys.exit(42)

# ToDo usedDatalog
