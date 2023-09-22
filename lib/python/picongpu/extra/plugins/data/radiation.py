# Copyright 2016-2023 Richard Pausch
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


class RadiationData:
    def __init__(self, filename, timestep):
        """
        Open references to an openPMD-api series or file to access radiation
        data.

        This constructor opens openPMD-api references to the radiation data
        and thus allows easy access to the complex amplitudes from the
        Lienard-Wiechert potential.

        Key-Argument:
        filename: string
                  path and name of the openPMD-api series or file
        timestep: int
                  PIC iteration to read radiation data from
        """
        # set openPMD-api file
        self.filename = filename
        self.rad_series = io.Series(filename, io.Access_Type.read_only)

        # extract time step
        self.timestep = timestep
        if (self.timestep not in self.rad_series.iterations):
            raise Exception("The selected timestep {} ".format(self.timestep)
                            + "is not available in the series.")
        self.iteration = self.rad_series.iterations[self.timestep]

        # Amplitude
        detectorAmplitude = self.iteration.meshes["Amplitude"]
        # A_x
        self.Ax_Re = detectorAmplitude["x_Re"][:, :, :]
        self.Ax_Im = detectorAmplitude["x_Im"][:, :, :]
        # A_y
        self.Ay_Re = detectorAmplitude["y_Re"][:, :, :]
        self.Ay_Im = detectorAmplitude["y_Im"][:, :, :]
        # A_z
        self.Az_Re = detectorAmplitude["z_Re"][:, :, :]
        self.Az_Im = detectorAmplitude["z_Im"][:, :, :]

        # conversion factor for spectra from PIC units to SI units
        # The value for unit_SI is consistent across all Amplitude datasets
        self.convert_to_SI = detectorAmplitude["x_Re"].unit_SI

        self.rad_series.flush()

    def get_Amplitude_x(self):
        """Returns the complex amplitudes in x-axis."""
        return ((self.Ax_Re[...] + 1j * self.Ax_Im[...])[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_y(self):
        """Returns the complex amplitudes in y-axis."""
        return ((self.Ay_Re[...] + 1j * self.Ay_Im[...])[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_z(self):
        """Returns the complex amplitudes in z-axis."""
        return ((self.Az_Re[...] + 1j * self.Az_Im[...])[:, :, 0] *
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

    def get_omega(self):
        """Returns frequency 'omega' of spectrum in [s^-1]."""
        omega_h = self.iteration.meshes["DetectorFrequency"]["omega"]
        omega = omega_h[0, :, 0]
        omega_unitSI = omega_h.unit_SI
        self.rad_series.flush()
        return omega * omega_unitSI

    def get_vector_n(self):
        """Returns the unit vector 'n' of the observation directions."""
        n_h = self.iteration.meshes["DetectorDirection"]
        n_x = n_h['x'][:, 0, 0]
        n_x_unitSI = n_h['x'].unit_SI

        n_y = n_h['y'][:, 0, 0]
        n_y_unitSI = n_h['y'].unit_SI

        n_z = n_h['z'][:, 0, 0]
        n_z_unit_SI = n_h['z'].unit_SI

        self.rad_series.flush()

        n_vec = np.empty((len(n_x), 3))
        n_vec[:, 0] = n_x * n_x_unitSI
        n_vec[:, 1] = n_y * n_y_unitSI
        n_vec[:, 2] = n_z * n_z_unit_SI
        return n_vec
