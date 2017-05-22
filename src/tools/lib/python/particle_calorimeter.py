"""
This file is part of PIConGPU.

PIConGPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PIConGPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PIConGPU.
If not, see <http://www.gnu.org/licenses/>.

Copyright 2017 Marco Garten, Axel Huebl
Authors: Axel Huebl
License: GPLv3+
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py


class ParticleCalorimeter:
    """
    A binned calorimeter of the amount of kinetic energy per solid angle and
    energy-per-particle. The solid angle bin is solely determined by the
    particle's momentum vector and not by its position, so we are emulating
    a calorimeter at infinite distance. The calorimeter takes into account all
    existing particles as well as optionally all particles which have already
    left the global simulation volume.
    """

    def __init__(self,
                 species_name="ph", sim="./",
                 period=1000, num_bins_yaw=64, num_bins_pitch=64,
                 num_bins_energy=1024, min_energy=10, max_energy=10000,
                 logscale=False, opening_yaw=360, opening_pitch=180,
                 pos_yaw=0, pos_pitch=0):
        """
        Parameters
        ----------
        species_name : string
            name of the species for which the particle calorimeter
            should be plotted, e.g. "ph"
        sim : string
            path to simOutput directory
        period : unsigned integer
            output periodicity of the histogram
        num_bins_yaw : unsigned integer
            number of bins resolving the yaw angle
        num_bins_pitch : unsigned integer
            number of bins resolving the pitch angle
        num_bins_energy : unsigned integer
            number of bins resolving the detected energies
        min_energy : float
            minimum detectable energy in keV
        max_energy : float
            maximum detectable energy in keV
        logscale : boolen
            if ``True`` the energy bins are logarithmically spaced
        opening_yaw : float
            slit opening in x in meters
        opening_pitch : float
            slit opening in z in meters
        pos_yaw : float
            yaw coordinate of the calorimeter position in degrees
            (default: +y-direction --> 0)
        pos_pitch : float
            pitch coordinate of the calorimeter position in degrees
            (default: +y-direction --> 0)
        """
        self.species_name = species_name
        self.sim = sim
        self.period = period
        self.num_bins_yaw = num_bins_yaw
        self.num_bins_pitch = num_bins_pitch
        self.num_bins_energy = num_bins_energy
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.logscale = logscale
        self.opening_yaw = opening_yaw
        self.opening_pitch = opening_pitch
        self.pos_yaw = pos_yaw
        self.pos_pitch = pos_pitch

        self.attribute_name = "calorimeter"

        self.energy_range = np.linspace(self.min_energy,
                                        self.max_energy,
                                        self.num_bins_energy)

        self.calo_data = np.zeros([self.num_bins_energy,
                                   self.num_bins_pitch,
                                   self.num_bins_yaw], dtype=np.float64)
        self.hist_data = np.zeros([self.num_bins_energy, ], dtype=np.float64)

        self.sim_unit = None

    def load_step(self,
                  step=5000):
        """
        Load an HDF5 File of a certain time step

        Parameters
        ----------
        time : unsigned integer
            PIConGPU output time step
        """
        # convert the time step into a string
        self.stepStr = str(int(step))

        self.calo_file = self.sim + "/simOutput/" \
            + self.species_name + "_" \
            + self.attribute_name + "/" \
            + self.species_name + "_" \
            + self.attribute_name + "_" \
            + self.stepStr + "_0_0_0.h5"

        if not os.path.isfile(self.calo_file):
            print(
                "Error: The output HDF5 file " +
                self.calo_file +
                " does not exist."
            )
            return

        f = h5py.File(self.calo_file, 'r')

        self.step = step
        # calorimeter data
        self.calo_data = np.array(f['/data/' +
                                    self.stepStr + "/" +
                                    self.attribute_name])
        # simulation unit
        self.sim_unit = f['/data/' +
                          self.stepStr + "/" +
                          self.attribute_name].attrs['unitSI']

        # cleanup
        f.close()

    def energy_histogram(self):
        """
        Creates an energy histogram that can be compared to the output of the
        `energyHistogram` plugin output. The data is therefore summed
        over the yaw and pitch angles.
        """
        self.hist_data = np.sum(self.calo_data, axis=(1, 2))

    def plot_histogram(self,
                       with_outliers=False,
                       norm=False):
        """
        Parameters
        ----------
        with_outliers : boolean
            plots also underflow and overflow bins if ``True``
        norm : boolean
            normalizes the histogram to the total number of counts
            in all bins if ``True``
        """
        # unit conversion: Joules to eV
        J2EV = 1.602e-19
        # unit conversion: simUnit to EV
        simUnitEV = self.sim_unit / J2EV

        figHist = plt.figure(0)
        axHist = figHist.add_subplot(111)

        # absolute histogram data [photon count]
        abs_hist_data = self.hist_data * \
            simUnitEV / 1e3 / self.energy_range
        energy_range = self.energy_range

        # remove outliers from data to plot
        if with_outliers is False:
            abs_hist_data = abs_hist_data[1:-1]
            energy_range = energy_range[1:-1]

        if norm is False:
            axHist.plot(energy_range,
                        abs_hist_data,
                        label="{}".format(int(self.step)))
        else:
            normed_hist_data = abs_hist_data / np.sum(abs_hist_data)
            axHist.plot(energy_range,
                        normed_hist_data,
                        label="{}".format(int(self.step)))

        axHist.set_xscale("log")
        axHist.set_yscale("log")
        axHist.set_xlabel("energy [keV]")
        axHist.set_ylabel("count [arb. u.]")
        axHist.set_title("`{}` energy histogram".format(self.species_name))
        axHist.set_xlim(.5 * self.min_energy, 2 * self.max_energy)
        axHist.legend(loc="best", title="counts after # steps")

        plt.draw()
        plt.show()

    def plot_calorimeter(self):
        """
        Plots an imshow of the particle calorimeter data.
        The data is summed over all energy bins and shown in units of keV.
        """
        # half the opening angles
        half_yaw = self.opening_yaw / 2
        half_pitch = self.opening_pitch / 2
        # solid angle "view range" of the calorimeter detector
        solid_angle_range = (-half_yaw + self.pos_yaw,
                             half_yaw + self.pos_yaw,
                             -half_pitch + self.pos_pitch,
                             half_pitch + self.pos_pitch)

        # unit conversion: Joules to eV
        J2EV = 1.602e-19
        # unit conversion: simUnit to EV
        simUnitEV = self.sim_unit / J2EV

        # data summed over the energy bins
        calo_sum = np.sum(self.calo_data, axis=0) * simUnitEV / 1e3

        figCal = plt.figure(1)
        axCal = figCal.add_subplot(111)

        caloPlot = axCal.imshow(
            calo_sum,
            extent=solid_angle_range,
            interpolation="None"
        )

        axCal.set_title(
            "energy calorimeter for `{}`".format(self.species_name)
        )
        axCal.set_xlabel("yaw angle [°]")
        axCal.set_ylabel("pitch angle [°]")
        cbar_label = r"$\Sigma E_\mathrm{ph} / \mathrm{pixel}$ [keV]"
        figCal.colorbar(caloPlot, label=cbar_label)
        plt.draw()
        plt.show()
