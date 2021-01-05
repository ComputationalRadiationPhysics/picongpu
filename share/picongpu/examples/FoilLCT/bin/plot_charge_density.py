#!/usr/bin/env python
#
"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""
import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(
    description='Compare the electron charge density of FoilLCT simulations'
)
parser.add_argument(
    'run_directory',
    metavar='D',
    type=str,
    nargs='+',
    help='path to the run directory of PIConGPU ' +
         '(the path before ``simOutput/``)'
)
parser.add_argument(
    '--step',
    type=int,
    default=2000,
    help='the time step'
)
parser.add_argument(
    '--sum',
    action='store_true',
    help='instead of free electron density, ' +
         'show summed charge density of all species'
)

args = parser.parse_args()

if len(args.run_directory) > 4:
    print("Error: Can only compare up to 4 directories!")
    exit(1)

sims = {}
for D in args.run_directory:
    run_name = os.path.basename(os.path.normpath(D))
    sims[run_name] = D

# note: assume fixed resolution of the simulation
dt = 4.91356e-18  # s
dx = 800.e-9 / 384.  # m
step = args.step

fig = plt.figure()
cax = fig.add_axes([0.87, 0.1, 0.01, 0.75])
ax1 = fig.add_axes([0.1, 0.5, 0.35, 0.35])
ax2 = fig.add_axes([0.5, 0.5, 0.35, 0.35])
ax3 = fig.add_axes([0.1, 0.1, 0.35, 0.35])
ax4 = fig.add_axes([0.5, 0.1, 0.35, 0.35])
axes = [ax1, ax2, ax3, ax4]


def get_nZ(flds, species):
    r = flds[species + "_chargeDensity"]
    d = r[()] * r.attrs["unitSI"] / 1.602e-19 * 1.e-6  # elements / cm^3
    return d


def plot_sim(ax, sim):
    f = h5.File(sims[sim] + "/simOutput/h5/simData_" + str(step) + ".h5", "r")
    ne = get_nZ(f["/data/" + str(step) + "/fields/"], "e")
    if args.sum:
        nH = get_nZ(f["/data/" + str(step) + "/fields/"], "H")
        nC = get_nZ(f["/data/" + str(step) + "/fields/"], "C")
        nN = get_nZ(f["/data/" + str(step) + "/fields/"], "N")
        d = ne + nH + nC + nN

    f.close()
    ax.set_title(sim)

    if args.sum:
        return ax.imshow(
            d,
            cmap='RdBu',
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=-3.e22,
            vmax=3.e22,
            extent=[0., dx * 1.e6 * d.shape[0], 0., dx * 1.e6 * d.shape[1]]
        )
    else:
        return ax.imshow(
            np.abs(ne),
            cmap='CMRmap_r',
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=0.,
            vmax=5.e23,
            extent=[0., dx * 1.e6 * ne.shape[0], 0., dx * 1.e6 * ne.shape[1]]
        )


i = 0
for sim in sims:
    print(sim)
    im = plot_sim(axes[i], sim)
    i += 1

if args.sum:
    fig.colorbar(im, cax=cax,
                 label=r'$n_{Z,\Sigma{e,H,C,N}}$ [$q_e \cdot$ cm$^{-3}$]')
else:
    fig.colorbar(im, cax=cax,
                 label=r'$n_e$ [$q_e \cdot$ cm$^{-3}$]')

fig.suptitle("time = {:5.3f} fs".format(dt * step * 1.e15))
ax3.set_xlabel(r'$x$ [$\mu$m]')
ax4.set_xlabel(r'$x$ [$\mu$m]')
ax1.set_ylabel(r'$y$ [$\mu$m]')
ax3.set_ylabel(r'$y$ [$\mu$m]')
plt.show()
