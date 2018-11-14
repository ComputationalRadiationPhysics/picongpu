#!/usr/bin/env python

"""
This file is part of PIConGPU.

It is supposed to give an estimate for the memory requirement of a PIConGPU
simulation per accelerator.

Copyright 2018 PIConGPU contributors
Authors: Marco Garten
License: GPLv3+
"""

import numpy as np


class PicongpuMemoryCalculator:
    """
    Memory requirement calculation tool for PIConGPU

    Contains calculation for fields, particles, random number generator
    and the calorimeter plugin. In-situ methods other than the calorimeter
    so far use up negligible amounts of memory on the accelerator.
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

        :param n_x: int
            number of cells in x direction (per accelerator)
        :param n_y: int
            number of cells in y direction (per accelerator)
        :param n_z: int
            number of cells in z direction (per accelerator)
        :param precision_bits:
            floating point precision for PIConGPU run
        """
        # local accelerator domain size
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
        Memory reserved for fields on each accelerator

        Parameters
        ----------

        n_x : int
            number of cells in x direction (per accelerator)
        n_y : int
            number of cells in y direction (per accelerator)
        n_z : int
            number of cells in z direction (per accelerator)
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
            required memory {unit: bytes} per accelerator
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

        # size of a data entry in bytes
        data_size = np.float32().itemsize
        # number of fields: 3 * 3 = x,y,z for E,B,J
        num_fields = 3 * 3 + field_tmp_slots
        # double buffer memory
        double_buffer_mem = double_buffer_cells * num_fields * data_size

        req_mem = data_size * num_fields * local_cells + double_buffer_mem
        return req_mem

    def mem_req_by_particles(
            self,
            target_n_x=None,
            target_n_y=None,
            target_n_z=None,
            num_additional_attributes=0,
            particles_per_cell=2
    ):
        """
        Memory reserved for all particles of a species on a GPU.
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

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per GPU and species
        """

        if target_n_x is None:
            target_n_x = self.n_x
        if target_n_y is None:
            target_n_y = self.n_y
        if target_n_z is None:
            target_n_z = self.n_z

        # memory required by the standard particle attributes
        standard_attribute_mem = np.array([
            3 * 4,  # momentum
            3 * 4,  # position
            1 * 8,  # multimask
            1 * 8,  # cell index in supercell (``lcellId_t``)
            1 * 8  # weighting
        ])

        # memory per particle for additional attributes {unit: byte}
        additional_mem = num_additional_attributes * 4
        # \TODO we assume 4 bytes here - check if that's really the case
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
    ):
        """
        Memory reserved for the random number generator state on each GPU.
        The RNG we use is: MRG32ka

        Parameters
        ----------
        n_x : int
            number of cells in x direction (per accelerator)
        n_y : int
            number of cells in y direction (per accelerator)
        n_z : int
            number of cells in z direction (per accelerator)

        Returns
        -------

        req_mem : int
            required memory {unit: bytes} per accelerator
        """
        if n_x is None:
            n_x = self.n_x
        if n_y is None:
            n_y = self.n_y
        if n_z is None:
            n_z = self.n_z

        req_mem_per_cell = 6 * 8  # bytes
        local_cells = n_x * n_y * n_z

        req_mem = req_mem_per_cell * local_cells
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
        each accelerator, once for particles in the simulation and once
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
            required memory {unit: bytes} per accelerator
        """
        if value_size is None:
            value_size = self.value_size

        req_mem_per_bin = value_size
        num_bins = n_energy * n_yaw * n_pitch
        req_mem = req_mem_per_bin * num_bins * 2

        return req_mem


if __name__ == "__main__":
    # THIS IS A USAGE EXAMPLE FOR AN NVIDIA P100 GPU with 16 GB MEMORY
    #
    # The target is a plastic foil.
    #
    # This is an estimate for how much memory is used per GPU if the whole
    # target would be fully ionized but does not move much. Of course the real
    # usage fluctuates and depends on the case. We use just one GPU out of the
    # whole group that simulates the full box and we take one that contains a
    # representative amount of the target.
    #

    cell_size = 4.402e-9  # m
    time_step = 8.474e-18  # s

    target_thickness = 1.852e-6  # m

    # number of cells per GPU
    Nx = 120
    Ny = 1136
    Nz = 76

    pmc = PicongpuMemoryCalculator(Nx, Ny, Nz)

    target_x = Nx
    target_y = np.int(target_thickness / cell_size)
    target_z = Nz

    # typical number of particles per cell which is multiplied later for
    # each species and its relative number of particles
    N_PPC = 4

    print("Memory requirement per GPU:")
    # field memory per GPU
    field_gpu = pmc.mem_req_by_fields(Nx, Ny, Nz, field_tmp_slots=2,
                                      particle_shape_order=3)
    print("+ fields: {:.2f} MB".format(
        field_gpu / (1024 * 1024)))
    # particle memory per GPU - only the target area contributes here
    e_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z, num_additional_attributes=0,
        # H,C,O preionization, 4 ppc
        particles_per_cell=3 * N_PPC) \
        + pmc.mem_req_by_particles(
            target_x, target_y, target_z, num_additional_attributes=0,
            # C and O are twice preionized
            particles_per_cell=N_PPC * ((6 - 2) + (8 - 2)))
    H_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=0,
        # no bound electrons since H is preionized
        particles_per_cell=N_PPC)
    C_gpu = pmc.mem_req_by_particles(target_x, target_y, target_z,
                                     num_additional_attributes=1,
                                     particles_per_cell=N_PPC)
    O_gpu = pmc.mem_req_by_particles(
        target_x, target_y, target_z,
        num_additional_attributes=1,
        particles_per_cell=N_PPC)
    # memory for calorimeters
    cal_gpu = pmc.mem_req_by_calorimeter(n_energy=1024, n_yaw=180, n_pitch=90
                                         ) * 2  # electrons and protons

    print("+ species:")
    print("- e: {:.2f} MB".format(e_gpu / (1024 * 1024)))
    print("- H: {:.2f} MB".format(H_gpu / (1024 * 1024)))
    print("- C: {:.2f} MB".format(C_gpu / (1024 * 1024)))
    print("- O: {:.2f} MB".format(O_gpu / (1024 * 1024)))
    rng_gpu = pmc.mem_req_by_rng(Nx, Ny, Nz)
    print("+ RNG states: {:.2f} MB".format(
        rng_gpu / (1024 * 1024)))
    print(
        "+ particle calorimeters: {:.2f} MB".format(
            cal_gpu / (1024 * 1024)))

    mem_sum = field_gpu + e_gpu + H_gpu + C_gpu + O_gpu + rng_gpu + cal_gpu
    print("Sum of required GPU memory: {:.2f} MB".format(
        mem_sum / (1024 * 1024)))
