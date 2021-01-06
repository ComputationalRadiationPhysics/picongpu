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

Example script for analyzing data from the *particleCalorimeter* plugin.
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
from particle_calorimeter import ParticleCalorimeter


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

times = np.linspace(1e3, 5e3, 5)
PC = ParticleCalorimeter(
    species_name="ph",
    sim="../",
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
    pos_pitch=0
)

for i in times:
    PC.load_step(step=i)
    PC.energy_histogram()
    PC.plot_histogram(norm=False)

PC.plot_calorimeter()
