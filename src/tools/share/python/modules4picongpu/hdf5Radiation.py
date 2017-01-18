# Copyright 2016-2017 Richard Pausch
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
import h5py


class radiationHDF5:
    def __init__(self, filename):
        """
        Open references to hdf5 file to access radiation data.

        This constructor opens h5py references to the radiation data
        and thus allows easy access to the complex amplitudes from the
        Lienard-Wiechert potential.

        Key-Argument:
        filename: string
                  path and name of the hdf5 radiation data file

        """
        # set hdf5 file
        self.filename = filename
        self.h5_file = h5py.File(filename, "r")
        # extract time step
        self.timestep = self.get_timestep()

        # Amplitude
        # A_x
        self.h5_Ax_Re = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/x/Re"]
        self.h5_Ax_Im = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/x/Im"]
        # A_y
        self.h5_Ay_Re = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/y/Re"]
        self.h5_Ay_Im = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/y/Im"]
        # A_z
        self.h5_Az_Re = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/z/Re"]
        self.h5_Az_Im = self.h5_file["/data/{}".format(self.timestep) +
                                     "/Amplitude/z/Im"]

        # conversion factor for spectra from PIC units to SI units
        self.convert_to_SI = self.h5_file["/data/{}".format(self.timestep) +
                                          "/Amplitude"].attrs['unitSI']

    def get_timestep(self):
        """Returns simulation timestep of the hdf5 data."""
        # this is a workaround till openPMD is implemented
        str_timestep = self.filename.split("_")[-4]
        if str_timestep.isdigit():
            return int(str_timestep)
        else:
            raise Exception("Could not extract timestep from " +
                            "filename (\"{}\") - ".format(self.filename) +
                            "Extracted: {}".format(str_timestep))

    def get_Amplitude_x(self):
        """Returns the complex amplitudes in x-axis."""
        return ((self.h5_Ax_Re.value + 1j * self.h5_Ax_Im.value)[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_y(self):
        """Returns the complex amplitudes in y-axis."""
        return ((self.h5_Ay_Re.value + 1j * self.h5_Ay_Im.value)[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_z(self):
        """Returns the complex amplitudes in z-axis."""
        return ((self.h5_Az_Re.value + 1j * self.h5_Az_Im.value)[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Spectra(self):
        """Returns real spectra in [Js]."""
        return (np.abs(self.get_Amplitude_x())**2 +
                np.abs(self.get_Amplitude_y())**2 +
                np.abs(self.get_Amplitude_z())**2)

    def get_Polarization_X(self):
        """Returns real spectra for x-polarization in [Js]."""
        return np.abs(self.get_Amplitude_x())**2

    def get_Polarization_Y(self):
        """Returns real spectra for y-polarization in [Js]."""
        return np.abs(self.get_Amplitude_y())**2

    def get_Polarization_Z(self):
        """Returns real spectra for z-polarization in [Js]."""
        return np.abs(self.get_Amplitude_z())**2
