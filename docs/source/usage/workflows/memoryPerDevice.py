#!/usr/bin/env python

"""
This file is part of PIConGPU.

Copyright 2018-2021 PIConGPU contributors
Authors: Marco Garten, Pawel Ordyna
License: GPLv3+
"""

from picongpu.utils import MemoryCalculator
from math import ceil

"""
@file

This file contains a usage example of the ``MemoryCalculator`` class
for our :ref:`FoilLCT example <usage-examples-foilLCT>` and its ``4.cfg``.

It is an estimate for how much memory is used per device if the whole
target would be fully ionized but does not move much. Of course the real
memory usage depends on the case and the dynamics inside the simulation.
We calculate the memory of just one device per row of GPUs in laser
propagation direction. We hereby assume that particles are distributed
equally in transverse direction, like it is set up in the FoilLCT example.
"""


cell_size = 0.8e-6 / 384.  # 2.083e-9 m
y0 = 0.5e-6  # position of foil front surface (m)
y1 = 1.5e-6  # position of foil rear surface (m)
L = 10e-9  # pre-plasma scale length (m)
L_cutoff = 4 * L  # pre-plasma length (m)

sim_dim = 2
# number of cells in the simulation
Nx_all, Ny_all, Nz_all = 256, 1280, 1
# number of GPU rows in each direction
x_rows, y_rows, z_rows = 2, 2, 1
# number of cells per GPU
Nx, Ny, Nz = Nx_all / x_rows, Ny_all / y_rows, Nz_all / z_rows

vacuum_cells = ceil((y0 - L_cutoff) / cell_size)  # in front of the target
# target cells (between surfaces + pre-plasma)
target_cells = ceil((y1 - y0 + 2 * L_cutoff) / cell_size)
# number of cells (y direction) on each GPU row
GPU_rows = [0] * y_rows
cells_to_spread = vacuum_cells + target_cells
# spread the cells on the GPUs
for ii, _ in enumerate(GPU_rows):
    if cells_to_spread >= Ny:
        GPU_rows[ii] = Ny
        cells_to_spread -= Ny
    else:
        GPU_rows[ii] = cells_to_spread
        break
# remove vacuum cells from the front rows
extra_cells = vacuum_cells
for ii, _ in enumerate(GPU_rows):
    if extra_cells >= Ny:
        GPU_rows[ii] = 0
        extra_cells -= Ny
    else:
        GPU_rows[ii] -= extra_cells
        break

pmc = MemoryCalculator(Nx, Ny, Nz)

# typical number of particles per cell which is multiplied later for
# each species and its relative number of particles
N_PPC = 6
# conversion factor to megabyte
megabyte = 1.0 / (1024 * 1024)

target_x = Nx  # full transverse dimension of the GPU
target_z = Nz


def sx(n):
    return {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20
                                           else int(str(n)[-1]), "th")


for row, target_y in enumerate(GPU_rows):
    print("{}{} row of GPUs:".format(row + 1, sx(row + 1)))
    print("* Memory requirement per GPU:")
    # field memory per GPU
    field_gpu = pmc.mem_req_by_fields(Nx, Ny, Nz, field_tmp_slots=2,
                                      particle_shape_order=2, sim_dim=sim_dim)
    print(" + fields: {:.2f} MB".format(
        field_gpu * megabyte))

    # electron macroparticles per supercell
    e_PPC = N_PPC * (
            # H,C,N pre-ionization - higher weighting electrons
            3
            # electrons created from C ionization
            + (6 - 2)
            # electrons created from N ionization
            + (7 - 2)
    )
    # particle memory per GPU - only the target area contributes here
    e_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=0,
        particles_per_cell=e_PPC
    )
    H_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        # no bound electrons since H is pre-ionized
        num_additional_attributes=0,
        particles_per_cell=N_PPC
    )
    C_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,  # number of bound electrons
        particles_per_cell=N_PPC
    )
    N_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,
        particles_per_cell=N_PPC
    )
    # memory for calorimeters
    cal_gpu = pmc.mem_req_by_calorimeter(
        n_energy=1024, n_yaw=360, n_pitch=1
    ) * 2  # electrons and protons
    # memory for random number generator states
    rng_gpu = pmc.mem_req_by_rng(Nx, Ny, Nz)

    print(" + species:")
    print("  - e: {:.2f} MB".format(e_gpu * megabyte))
    print("  - H: {:.2f} MB".format(H_gpu * megabyte))
    print("  - C: {:.2f} MB".format(C_gpu * megabyte))
    print("  - N: {:.2f} MB".format(N_gpu * megabyte))
    print(" + RNG states: {:.2f} MB".format(
        rng_gpu * megabyte))
    print(
        " + particle calorimeters: {:.2f} MB".format(
            cal_gpu * megabyte))

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + N_gpu + rng_gpu + cal_gpu
    print("* Sum of required GPU memory: {:.2f} MB".format(
        mem_sum * megabyte))
