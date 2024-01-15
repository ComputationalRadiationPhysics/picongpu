# Copyright 2023 Richard Pausch
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
import numpy as np
import openpmd_api as io


class particleCalorimeter:
    """
    class to read data from the PIConGPU particle calorimeter plugin

    It provides various methods to get data.
    It also provides a dict `detector_params` with all meta-data.
    """

    def __init__(self, seriesFilename):
        """
        initalize class object and extract parameters

        Parameters
        ----------
        seriesFilename : string
                         path and openPMD file format for input series
                         e.g. "simOutput/e_calorimeter/e_calorimeter_all_%T.bp"
        """
        self.series = io.Series(seriesFilename, access=io.Access_Type.read_only)
        self.iterations = list(self.series.iterations)

        # get first iteration to extrat meta information
        for time_iteration in self.iterations:
            it = self.series.iterations[time_iteration]
            break

        self.detector_params = {}
        h = it.meshes["calorimeter"][io.Mesh_Record_Component.SCALAR]
        for i in h.attributes:
            self.detector_params[i] = h.get_attribute(i)

        self.detector_params["N_yaw"] = h.shape[-1]
        self.detector_params["N_pitch"] = h.shape[-2]

        if len(h.shape) == 3:
            self.detector_params["N_energy"] = h.shape[-3]
        else:
            self.detector_params["N_energy"] = None

    def getIterations(self):
        """
        list of all available iterations

        Returns
        -------
        it : list of integers [PIC iterations]
        """
        return self.iterations

    def getYaw(self):
        """
        returns array of yaw values in degree

        Returns
        -------
        yaw : ndarray [degree]
        """
        return (
            np.linspace(
                -self.detector_params["maxYaw[deg]"],
                +self.detector_params["maxYaw[deg]"],
                self.detector_params["N_yaw"],
            )
            + self.detector_params["posYaw[deg]"]
        )

    def getPitch(self):
        """
        returns array of pitch values in degree

        Returns
        -------
        pitch : ndarray [degree]
        """
        return (
            np.linspace(
                -self.detector_params["maxPitch[deg]"],
                +self.detector_params["maxPitch[deg]"],
                self.detector_params["N_pitch"],
            )
            + self.detector_params["posPitch[deg]"]
        )

    def getEnergy(self):
        """
        returns array of energy bin values in keV
        if no energy binning was used, None is returned

        Returns
        -------
        energy : ndarray or None [keV]
        """
        if self.detector_params["N_energy"] is None:
            return None
        else:
            if self.detector_params["logScale"] is False:
                return np.linspace(
                    self.detector_params["minEnergy[keV]"],
                    self.detector_params["maxEnergy[keV]"],
                    self.detector_params["N_energy"],
                )
            else:
                return np.logspace(
                    np.log10(self.detector_params["minEnergy[keV]"]),
                    np.log10(self.detector_params["maxEnergy[keV]"]),
                    self.detector_params["N_energy"],
                )

    def getData(self, iteration):
        """
        returns array of calorimeter data from
        specified iteration. The values are the total energy
        per bin in Joule.

        Parameters
        ----------
        iteration : int
                    iteration number to read data from

        Returns
        -------
        energy : ndarray [Joule]
                 either 2D array (without energy binning)
                 or 3D array (with energy binning)
                 size: [N_energy, N_pitch, N_yaw]
        """
        it = self.series.iterations[iteration]
        h = it.meshes["calorimeter"][io.Mesh_Record_Component.SCALAR]
        data = h.load_chunk()
        self.series.flush()
        return data * self.detector_params["unitSI"]
