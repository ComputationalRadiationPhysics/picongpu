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

Example script for analyzing data from the *energyHistogram* plugin.
If the file is just executed in place it will plot the photon energy
histogram of the default Bremsstrahlung example.
There will be 5 datasets for the 5 different output iterations. The plots
will also not contain the outliers.

Copyright 2017-2021 Marco Garten, Axel Huebl
Authors: Axel Huebl
License: GPLv3+
"""
import matplotlib as mpl
import numpy as np
from energy_histogram import EnergyHistogram


# custom matplotlib settings
params = {
    'font.size': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.figsize': [12, 8],
    'legend.fontsize': 20,
    'legend.frameon': "False",
    'legend.markerscale': 2,
    'legend.numpoints': 1,
    'lines.linestyle': '',
    'lines.marker': "+",
    'lines.markeredgewidth': 2,
    'lines.markersize': 6
}

# overwrite matplotlib defaults
mpl.rcParams.update(params)


# simulation time steps
times = np.linspace(1e3, 5e3, 5)
hist = EnergyHistogram(
    species_name="ph",
    sim="../",
    period=1000,
    bin_count=1024,
    min_energy=10,
    max_energy=10000,
    distance_to_detector=0,
    slit_detector_x=None,
    slit_detector_z=None
)

for i in times:
    hist.plot(time=int(i))
