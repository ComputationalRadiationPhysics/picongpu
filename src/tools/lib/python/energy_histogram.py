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


class EnergyHistogram:
    """
    Class to help plotting data fromt the energyHistogram plugin.
    """
    def __init__(
            self,
            species_name="ph",
            sim="./",
            period=1000,
            bin_count=1024,
            min_energy=10,
            max_energy=10000,
            distance_to_detector=0,
            slit_detector_x=None,
            slit_detector_z=None):
        """
        Parameters
        ----------
        species_name : string
            name of the species for which the energy histogram should be
            plotted, e.g. "ph"
        sim : string
            path to simOutput directory
        period : unsigned integer
            output periodicity of the histogram
        bin_count : unsigned integer
            energy resolution of the histogram (number of bins)
        min_energy : float
            minimum detectable energy in keV
        max_energy : float
            maximum detectable energy in keV
        distance_to_detector : float
            distance to detector in y-direction in meters (if == 0: all
            particles are considered)
        slit_detector_x : float
            slit opening in x in meters
        slit_detector_z : float
            slit opening in z in meters
        """
        self.species_name = species_name
        self.sim = sim
        self.period = period
        self.bin_count = bin_count
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.distance_to_detector = distance_to_detector
        self.slit_detector_x = slit_detector_x
        self.slit_detector_z = slit_detector_z

        self.histFile = "{}/simOutput/{}_energyHistogram.dat".format(
            sim, species_name)
        # matrix with data in keV
        #   note: skips columns for iteration step, underflow bin (first 2) and
        #         overflow bin, summation of all bins (last two)
        self.histData = np.loadtxt(self.histFile)[:, 2:-2]

        # upper range of each bin in keV
        # note: only reads first row and selects the 1024 bins (default, see
        # binCount) in columns 2:1026
        self.bins = np.loadtxt(
            self.histFile,
            comments='',
            usecols=range(
                2,
                2 + self.bin_count)
            )[0, :]

        # cumulates all counts of photons that have energies below
        # ``min_energy``
        self.underflow_bins = np.loadtxt(self.histFile)[:, 1]
        # cumulates all counts of photons that have energies above
        # ``max_energy``
        self.overflow_bins = np.loadtxt(self.histFile)[:, -2]
        # simulation time steps for output
        self.time_steps = np.loadtxt(self.histFile)[:, 0]
        # cumulative sum of all counts in all bins for each output time step
        self.total_counts = np.loadtxt(self.histFile)[:, -1]

    def plot(self,
             time=5000,
             with_outliers=False,
             norm=False):
        """
        time : unsigned integer
            simulation time step for which the histogram is plotted
        with_outliers : boolean
            plots also underflow and overflow bins if ``True``
        norm : boolean
            normalizes the histogram to the total number of counts
            in all bins if ``True``
        """
        iteration = int(time / self.period)

        figEHist = plt.figure(0)
        subEHist = figEHist.add_subplot(111)

        if (norm):
            # plot the bins normalized to the total number of counts in this
            # iteration
            EHist = subEHist.plot(
                self.bins,
                self.histData[iteration] / self.total_counts[iteration],
                label=time
            )
        else:
            # plot the histogram without normalization
            EHist = subEHist.plot(
                self.bins,
                self.histData[iteration],
                label=time)

        subEHist.set_xscale("log")
        subEHist.set_yscale("log")
        subEHist.set_xlabel("energy [keV]")
        subEHist.set_ylabel("count [arb. u.]")
        subEHist.set_title("`{}` energy histogram".format(self.species_name))

        if with_outliers:
            if norm:
                EHistUnderFlow = subEHist.plot(
                    self.min_energy,
                    self.underflow_bins[iteration] /
                    self.total_counts[iteration],
                    marker='+',
                    ms=6,
                    ls='',
                    mew=2,
                    label="underflow bin")
                EHistOverFlow = subEHist.plot(
                    self.max_energy,
                    self.overflow_bins[iteration] /
                    self.total_counts[iteration],
                    marker='+',
                    ms=6,
                    ls='',
                    mew=2,
                    label="overflow bin")
            else:
                EHistUnderFlow = subEHist.plot(
                    self.min_energy,
                    self.underflow_bins[iteration],
                    marker='+',
                    ms=6,
                    ls='',
                    mew=2,
                    label="underflow bin"
                )
                EHistOverFlow = subEHist.plot(
                    self.max_energy,
                    self.overflow_bins[iteration],
                    marker='+',
                    ms=6,
                    ls='',
                    mew=2,
                    label="overflow bin"
                )

        subEHist.set_xlim(.5 * self.min_energy, 2 * self.max_energy)

        subEHist.legend(loc="best", title="counts after # steps")
