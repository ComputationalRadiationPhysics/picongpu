#!/usr/bin/env python

"""
This file is part of PIConGPU.

Copyright 2018-2019 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""

from picongpu.utils import MemoryCalculator

"""
@file

This file contains a usage example of the ``MemoryCalculator`` class
for our :ref:`FoilLCT example <usage-examples-foilLCT>` and its ``4.cfg``.

It is an estimate for how much memory is used per device if the whole
target would be fully ionized but does not move much. Of course the real
memory usage depends on the case and the dynamics inside the simulation.
We calculate the memory of just one device out of the whole group that
simulates the full box and we take one that we expect to experience the
maximum memory load due to hosting a large part of the target.
"""


cell_size = 0.8e-6 / 384.  # 2.083e-9 m

y0 = 0.5e-6  # position of foil surface (m)
y1 = 1.0e-6  # target thickness (m)
L = 10.e-9  # pre-plasma scale length (m)
L_cutoff = 4.0 * L  # pre-plasma length (m)

# number of cells per device
Nx = 128
Ny = 640
Nz = 1

vacuum_cells = (y0 - L_cutoff) / cell_size  # with pre-plasma: 221 cells
target_cells = (y1 - y0 + 2 * L_cutoff) / cell_size  # 398 cells

pmc = MemoryCalculator(Nx, Ny, Nz)

target_x = Nx  # full transversal dimension of the device
target_y = target_cells  # only the first row of devices holds the target
target_z = Nz

# typical number of particles per cell which is multiplied later for
# each species and its relative number of particles
N_PPC = 6

# conversion factor to megabyte
megabyte = 1.0 / (1024 * 1024)

print("Memory requirement per device:")
# field memory per device
field_device = pmc.mem_req_by_fields(Nx, Ny, Nz, field_tmp_slots=2,
                                     particle_shape_order=2)
print("+ fields: {:.2f} MB".format(
      field_device * megabyte))

# electron macroparticles per supercell
e_PPC = N_PPC * (
    # H,C,N pre-ionization - higher weighting electrons
    3 \
    # electrons created from C ionization
    + (6 - 2) \
    # electrons created from N ionization
    + (7 - 2)
)
# particle memory per device - only the target area contributes here
e_device = pmc.mem_req_by_particles(
    target_x, target_y, target_z,
    num_additional_attributes=0,
    particles_per_cell=e_PPC
)
H_device = pmc.mem_req_by_particles(
    target_x, target_y, target_z,
    # no bound electrons since H is preionized
    num_additional_attributes=0,
    particles_per_cell=N_PPC
)
C_device = pmc.mem_req_by_particles(
    target_x, target_y, target_z,
    num_additional_attributes=1,
    particles_per_cell=N_PPC
)
N_device = pmc.mem_req_by_particles(
    target_x, target_y, target_z,
    num_additional_attributes=1,
    particles_per_cell=N_PPC
)
# memory for calorimeters
cal_device = pmc.mem_req_by_calorimeter(
    n_energy=1024, n_yaw=360, n_pitch=1
) * 2  # electrons and protons
# memory for random number generator states
rng_device = pmc.mem_req_by_rng(Nx, Ny, Nz)

print("+ species:")
print("- e: {:.2f} MB".format(e_device * megabyte))
print("- H: {:.2f} MB".format(H_device * megabyte))
print("- C: {:.2f} MB".format(C_device * megabyte))
print("- N: {:.2f} MB".format(N_device * megabyte))
print("+ RNG states: {:.2f} MB".format(
    rng_device * megabyte))
print(
    "+ particle calorimeters: {:.2f} MB".format(
        cal_device * megabyte))

mem_sum = (field_device + e_device + H_device + C_device +
           N_device + rng_device + cal_device)
print("Required memory per device: {:.2f} MB".format(
    mem_sum * megabyte))
