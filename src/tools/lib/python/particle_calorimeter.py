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
from scipy import constants as sc
import h5py
import os


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
                 species_name="e",
                 species_filter="",
                 sim="./",
                 period=1000,
                 num_bins_yaw=64,
                 num_bins_pitch=64,
                 num_bins_energy=1024,
                 min_energy=10,
                 max_energy=10000,
                 logscale=False,
                 opening_yaw=360,
                 opening_pitch=180,
                 pos_yaw=0,
                 pos_pitch=0):
        """
        Parameters
        ----------
        species_name : string
            name of the species for which the particle calorimeter
            should be plotted (default: "e")
        species_filter : string
            file suffix for plugin output created by a run over a subselection
            of the species according to an optional filter argument
            (default: "")
        sim : string
            path to simOutput directory (default: "./")
        period : unsigned integer
            output periodicity of the histogram (default: 1000)
        num_bins_yaw : unsigned integer
            number of bins resolving the yaw angle (default: 64)
        num_bins_pitch : unsigned integer
            number of bins resolving the pitch angle (default: 64)
        num_bins_energy : unsigned integer
            number of bins resolving the detected energies (default: 1024)
        min_energy : float
            minimum detectable energy in keV (default: 10)
        max_energy : float
            maximum detectable energy in keV (default: 10000)
        logscale : boolean
            if ``True`` the energy bins are logarithmically spaced
            (default: False)
        opening_yaw : float
            opening yaw angle in degrees (default: 360)
        opening_pitch : float
            opening pitch angle in degrees (default: 180)
        pos_yaw : float
            yaw coordinate of the calorimeter position in degrees
            (default: +y-direction --> 0)
        pos_pitch : float
            pitch coordinate of the calorimeter position in degrees
            (default: +y-direction --> 0)
        """
        self.species_name = species_name
        self.species_filter = species_filter
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

        self.step = 0
        self.stepStr = str(self.step)

        self.attribute_name = "calorimeter"

        if self.logscale == True:
            self.energy_range = np.logspace(np.log10(self.min_energy),
                                        np.log10(self.max_energy),
                                        self.num_bins_energy)
        else:
            self.energy_range = np.linspace(self.min_energy,
                                        self.max_energy,
                                        self.num_bins_energy)

        self.calo_data = np.zeros([self.num_bins_energy,
                                   self.num_bins_pitch,
                                   self.num_bins_yaw], dtype=np.float64)
        self.hist_data = np.zeros([self.num_bins_energy, ], dtype=np.float64)

        self.sim_unit = None

    def load_step(self,
                  step=0):
        """
        Load an HDF5 file of a certain time step

        Parameters
        ----------
        step : unsigned integer
            PIConGPU output time step (default: 0)
        """
        # convert the time step into a string
        self.stepStr = str(int(step))

        filter_fname = ""
        if self.species_filter != "":
            filter_fname = "_"+ species_filter

        self.calo_file = "/{}/simOutput/{}{}_{}/{}_{}_{}_0_0_0.h5".format(
            self.sim,
            self.species_name,
            self.species_filter,
            self.attribute_name,
            self.species_name,
            self.attribute_name,
            self.stepStr
        )

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

        self.calo_data = np.array(f['/data/{}/{}'.format(
                            self.stepStr,
                            self.attribute_name
                            )])
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
        Plots an energy histogram. Requires that the function
        ``energy_histogram`` has been called before.

        Parameters
        ----------
        with_outliers : boolean
            plots also underflow and overflow bins if ``True``
            (default: False)
        norm : boolean
            normalizes the histogram to the total number of counts
            in all bins if ``True``
            (default: False)
        """
        # check if `hist_data` is even filled with values to prevent a long
        # error message
        if np.sum(self.hist_data) <= 0:
            print(
                "There is no valid data for the histogram. "+
                "You may have forgotten to execute `energy_histogram` first."
            )
            return

        # unit conversion: Joules to eV
        J2EV = sc.electron_volt
        # unit conversion: simUnit to EV
        simUnitEV = self.sim_unit / J2EV
        # unit conversion factor: 1 eV in keV
        EV2KEV = 1e-3

        figHist = plt.figure(0)
        axHist = figHist.add_subplot(111)

        # absolute histogram data [photon count]
        abs_hist_data = self.hist_data * \
            simUnitEV * EV2KEV / self.energy_range
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
        J2EV = sc.electron_volt
        # unit conversion: simUnit to EV
        simUnitEV = self.sim_unit / J2EV
        # unit conversion factor: 1 eV in keV
        EV2KEV = 1e-3

        # data summed over the energy bins
        calo_sum = np.sum(self.calo_data, axis=0) * simUnitEV * EV2KEV

        figCal = plt.figure(1)
        axCal = figCal.gca()

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
