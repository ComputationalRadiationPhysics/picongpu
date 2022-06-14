"""
This file is part of PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

from TestKHIManager import TestKHIManager
from time import localtime, strftime


class TestKHI(TestKHIManager):
    """
    this class is primarily used to create the log files for the test suite

    funtions:
    -------
    createResultLog(path: str)
         create the file result.log under the given directory
    """

    def createResultLog(self, path: str):
        """
        create the file result.log under the given directory
        this file contains a few data of the simulation, like the growth rate
        and the difference, but also the test result

        input:
        -------
        path:   str
                path, where file should be saved
        """

        lt = localtime()
        date = strftime("date: %d.%m.%Y", lt)
        timeOfDay = strftime("time: %H:%M:%S", lt)

        resultFile = open(path + "/result.log", "w")

        # write the header
        resultFile.write(date + " " + timeOfDay + "\n\n")

        # write the result from the test
        if self.getTestResult():
            resultFile.write("Test was successful with the following" +
                             " results: \n\n")
            result = 0
        else:
            resultFile.write("Test failed with the following results: \n\n")
            result = 1
        # write the data from the growth rates
        resultFile.write("Predominant regime of KHI and the theoretical" +
                         " maximum growth rate: \n")
        resultFile.write("    " + self.getMaxGrowthRate(False)[0] + "\n")
        resultFile.write("Growth rate calculated from the simulation: " +
                         "{} \n\n".format(self.getMaxGrowthRate_Sim()))

        # acceptanceLevel
        resultFile.write("an acceptance range in the growth rate was" +
                         " considered from: \n\n")
        resultFile.write("          {}           \n\n".format(
                                   self.getAcceptanceRange()))

        # Difference
        resultFile.write("This corresponds to an inaccuracy of: \n\n")
        resultFile.write("Difference: {} \n".format(self.getMaxDifference()))
        resultFile.write("Difference in percentage: {} % \n".format(
                               self.getMaxDifference(percentage=True)))

        resultFile.close()

        return result
