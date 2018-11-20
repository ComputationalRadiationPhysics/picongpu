#!/usr/bin/env python

"""
This file is part of PIConGPU.

It is supposed to give an estimate for the memory requirement of a PIConGPU
simulation per device.

Copyright 2018 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""

import numpy as np


class MemoryCalculator:
    """
    Memory requirement calculation tool for PIConGPU

    Contains calculation for fields, particles, random number generator
    and the calorimeter plugin. In-situ methods other than the calorimeter
    so far use up negligible amounts of memory on the device.
    """

    def __init__(
            self,
            n_x,
            n_y,
            n_z,
            precision_bits=32
    ):
        """
        Class constructor

        Parameters
        ----------

        n_x : int
            number of cells in x direction (per device)
        n_y : int
            number of cells in y direction (per device)
        n_z : int
            number of cells in z direction (per device)
        precision_bits : int
            floating point precision for PIConGPU run
        """
        # local device domain size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        self.precision_bits = precision_bits

        if self.precision_bits == 32:
            # value size in bytes
            self.value_size = np.float32().itemsize
        elif self.precision_bits == 64:
            # value size in bytes
            self.value_size = np.float64().itemsize
        else:
            raise ValueError(
                "PIConGPU only supports either 32 or 64 bits precision."
            )

    def mem_req_by_fields(
            self,
            n_x=None,
            n_y=None,
            n_z=None,
            field_tmp_slots=1,
            particle_shape_order=2,
            sim_dim=3
    ):
        """
        Memory reserved for fields on each device

        Parameters
        ----------

        n_x : int
            number of cells in x direction (per device)
        n_y : int
            number of cells in y direction (per device)
        n_z : int
            number of cells in z direction (per device)
        field_tmp_slots : int
            number of slots for temporary fields
            (see PIConGPU ``memory.param`` : ``fieldTmpNumSlots``)
        particle_shape_order : int
            numerical order of the particle shape (see PIConGPU
            ``species.param``: e.g. ``particles::shapes::PCS : 3rd order``)
        sim_dim : int
            simulation dimension (available for PIConGPU: 2 and 3)

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device
        """
        if n_x is None:
            n_x = self.n_x
        if n_y is None:
            n_y = self.n_y
        if n_z is None:
            n_z = self.n_z

        # guard size in super cells in x, y, z
        guard_size_supercells = np.array([1, 1, 1])

        pso = particle_shape_order

        if sim_dim == 2:
            # super cell size in cells in x, y, z
            supercell_size = np.array(
                [16, 16, 1])  # \TODO make this more generic
            local_cells = (n_x + supercell_size[0] * 2 * guard_size_supercells[
                0]) * (n_y + supercell_size[1] * 2 * guard_size_supercells[1])

            # cells around core-border region due to particle shape
            double_buffer_cells = (n_x + pso) * (n_y + pso) - n_x * n_y
        elif sim_dim == 3:
            # super cell size in cells in x, y, z
            # \TODO make this more generic
            supercell_size = np.array([8, 8, 4])
            local_cells = (
                n_x + supercell_size[0] * 2 *
                guard_size_supercells[0]) \
                * (n_y + supercell_size[1] * 2 *
                   guard_size_supercells[1]) \
                * (n_z + supercell_size[2] * 2 *
                   guard_size_supercells[2])

            # cells around core-border region due to particle shape
            double_buffer_cells = (n_x + pso) * (n_y + pso) * (n_z + pso) \
                - n_x * n_y * n_z
        else:
            raise ValueError(
                "PIConGPU only runs in either 2D or 3D: ",
                sim_dim,
                " =/= {2, 3}")

        # number of fields: 3 * 3 = x,y,z for E,B,J
        num_fields = 3 * 3 + field_tmp_slots
        # double buffer memory
        double_buffer_mem = double_buffer_cells * num_fields * self.value_size

        req_mem = self.value_size * num_fields * local_cells \
            + double_buffer_mem
        return req_mem

    def mem_req_by_particles(
            self,
            target_n_x=None,
            target_n_y=None,
            target_n_z=None,
            num_additional_attributes=0,
            particles_per_cell=2,
            sim_dim=3
    ):
        """
        Memory reserved for all particles of a species on a device.
        We currently neglect the constant species memory.

        Parameters
        ----------

        target_n_x : int
            number of cells in x direction containing the target
        target_n_y : int
            number of cells in y direction containing the target
        target_n_z : int
            number of cells in z direction containing the target
        num_additional_attributes : int
            number of additional attributes like e.g. ``boundElectrons``
        particles_per_cell : int
            number of particles of the species per cell
        sim_dim : int
            simulation dimension (available for PIConGPU: 2 and 3)

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device and species
        """

        if target_n_x is None:
            target_n_x = self.n_x
        if target_n_y is None:
            target_n_y = self.n_y
        if target_n_z is None:
            target_n_z = self.n_z

        # memory required by the standard particle attributes
        standard_attribute_mem = np.array([
            3 * self.value_size,  # momentum
            sim_dim * self.value_size,  # position
            1,  # multimask (``uint8_t``)
            2,  # cell index in supercell (``typedef uint16_t lcellId_t``)
            1 * self.value_size  # weighting
        ])

        # memory per particle for additional attributes {unit: byte}
        additional_mem = num_additional_attributes * self.value_size
        # \TODO we assume value_size here - that could be different
        # cells filled by the target species
        local_cells = target_n_x * target_n_y * target_n_z

        req_mem = local_cells * (np.sum(
            standard_attribute_mem) + additional_mem) * particles_per_cell
        return req_mem

    def mem_req_by_rng(
            self,
            n_x=None,
            n_y=None,
            n_z=None,
            generator_method="XorMin"
    ):
        """
        Memory reserved for the random number generator state on each device.

        Check ``random.param`` for a choice of random number generators.
        If you find that your required RNG state is large (> 300 MB) please see
        ``memory.param`` for a possible adjustment of the
        ``reservedGpuMemorySize``.

        Parameters
        ----------
        n_x : int
            number of cells in x direction (per device)
        n_y : int
            number of cells in y direction (per device)
        n_z : int
            number of cells in z direction (per device)
        generator_method : str
            random number generator method - influences the state size per cell
            possible options: "XorMin", "MRG32k3aMin", "AlpakaRand"
            - (GPU default: "XorMin")
            - (CPU default: "AlpakaRand")

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device
        """
        if n_x is None:
            n_x = self.n_x
        if n_y is None:
            n_y = self.n_y
        if n_z is None:
            n_z = self.n_z

        if generator_method == "XorMin":
            state_size_per_cell = 6 * 4  # bytes
        elif generator_method == "MRG32k3aMin":
            state_size_per_cell = 6 * 8  # bytes
        elif generator_method == "AlpakaRand":
            state_size_per_cell = 7 * 4  # bytes
        else:
            raise ValueError(
                "{} is not an available RNG for PIConGPU.".format(
                    generator_method
                ), "Please choose one of the following: ",
                "'XorMin', 'MRG32k3aMin', 'AlpakaRand'"
            )

        # CORE + BORDER region of the device, GUARD currently has no RNG state
        local_cells = n_x * n_y * n_z

        req_mem = state_size_per_cell * local_cells
        return req_mem

    def mem_req_by_calorimeter(
            self,
            n_energy,
            n_yaw,
            n_pitch,
            value_size=None
    ):
        """
        Memory required by the particle calorimeter plugin.
        Each of the (``n_energy`` x ``n_yaw`` x ``n_pitch``) bins requires
        a value (32/64 bits). The whole calorimeter is represented twice on
        each device, once for particles in the simulation and once
        for the particles that leave the box.

        Parameters
        ----------

        n_energy : int
            number of bins on the energy axis
        n_yaw : int
            number of bins for the yaw angle
        n_pitch : int
            number of bins for the pitch angle
        value_size : int
            value size in particle calorimeter {unit: byte} (default: 4)

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per device
        """
        if value_size is None:
            value_size = self.value_size

        req_mem_per_bin = value_size
        num_bins = n_energy * n_yaw * n_pitch
        # one calorimeter instance for particles in the box
        # another calorimeter instance for particles leaving the box
        # makes a factor of 2 for the required memory
        req_mem = req_mem_per_bin * num_bins * 2

        return req_mem


if __name__ == "__main__":
    """
    This is a usage example of the ``MemoryCalculator`` class
    for our :ref:`FoilLCT example <usage-examples-foilLCT>` and its ``4.cfg``.

    This is an estimate for how much memory is used per GPU if the whole
    target would be fully ionized but does not move much. Of course the real
    memory usage depends on the case and the dynamics inside the simulation.
    We calculate the memory just one GPU out of the whole group that simulates
    the full box and we take one that we expect to experience the maximum
    memory load due to hosting a large part of the target.
    """

    print("This is a usage example of the 'MemoryCalculator' class \n",
          "using our 'FoilLCT' example and its '4.cfg' configuration file. \n")

    cell_size = 0.8e-6 / 384.  # 2.083e-9 m

    y0 = 0.5e-6  # position of foil surface (m)
    y1 = 1.0e-6  # target thickness (m)
    L = 10.e-9  # pre-plasma scale length (m)
    L_cutoff = 4.0 * L  # pre-plasma length (m)

    # number of cells per GPU
    Nx = 128
    Ny = 640
    Nz = 1

    vacuum_cells = (y0 - L_cutoff) / cell_size  # with pre-plasma: 221 cells
    target_cells = (y1 - y0 + 2 * L_cutoff) / cell_size  # 398 cells

    pmc = MemoryCalculator(Nx, Ny, Nz)

    target_x = Nx  # full transversal dimension of the GPU
    target_y = target_cells  # only the first row of GPUs holds the target
    target_z = Nz

    # typical number of particles per cell which is multiplied later for
    # each species and its relative number of particles
    N_PPC = 6

    print("Memory requirement per GPU:")
    # field memory per GPU
    field_gpu = pmc.mem_req_by_fields(Nx, Ny, Nz, field_tmp_slots=2,
                                      particle_shape_order=2)
    print("+ fields: {:.2f} MB".format(
        field_gpu / (1024 * 1024)))

    # electron macroparticles per supercell
    e_PPC = N_PPC * (
        # H,C,N pre-ionization - higher weighting electrons
        3 \
        # electrons created from C ionization
        + (6 - 2) \
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
        # no bound electrons since H is preionized
        num_additional_attributes=0,
        particles_per_cell=N_PPC
    )
    C_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,
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

    print("+ species:")
    print("- e: {:.2f} MB".format(e_gpu / (1024 * 1024)))
    print("- H: {:.2f} MB".format(H_gpu / (1024 * 1024)))
    print("- C: {:.2f} MB".format(C_gpu / (1024 * 1024)))
    print("- N: {:.2f} MB".format(N_gpu / (1024 * 1024)))
    print("+ RNG states: {:.2f} MB".format(
        rng_gpu / (1024 * 1024)))
    print(
        "+ particle calorimeters: {:.2f} MB".format(
            cal_gpu / (1024 * 1024)))

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + N_gpu + rng_gpu + cal_gpu
    print("Sum of required GPU memory: {:.2f} MB".format(
        mem_sum / (1024 * 1024)))
