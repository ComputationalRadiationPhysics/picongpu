#!/usr/bin/env python

"""
This file is part of PIConGPU.

It is supposed to give an estimate for the memory requirement of a PIConGPU
simulation per device.

Copyright 2018-2021 PIConGPU contributors
Authors: Marco Garten, Sergei Bastrakov
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
            sim_dim=3,
            pml_n_x=0,
            pml_n_y=0,
            pml_n_z=0
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
        pml_n_x : int
            number of PML cells in x direction, combined for both sides
        pml_n_y : int
            number of PML cells in y direction, combined for both sides
        pml_n_z : int
            number of PML cells in z direction, combined for both sides

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
        # PML size cannot exceed the local grid size
        pml_n_x = min(pml_n_x, n_x)
        pml_n_y = min(pml_n_y, n_y)
        pml_n_z = min(pml_n_z, n_z)

        # guard size in super cells in x, y, z
        guard_size_supercells = np.array([1, 1, 1])

        pso = particle_shape_order

        if sim_dim == 2:
            # super cell size in cells in x, y, z
            supercell_size = np.array(
                [16, 16, 1])  # \TODO make this more generic
            local_cells = (n_x + supercell_size[0] * 2 * guard_size_supercells[
                0]) * (n_y + supercell_size[1] * 2 * guard_size_supercells[1])
            local_pml_cells = n_x * n_y - (n_x - pml_n_x) * (n_y - pml_n_y)

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
            local_pml_cells = n_x * n_y * n_z \
                - (n_x - pml_n_x) * (n_y - pml_n_y) * (n_z - pml_n_z)

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
        # number of additional PML field components: when enabled,
        # 2 additional scalar fields for each of Ex, Ey, Ez, Bx, By, Bz
        num_pml_fields = 12

        req_mem = self.value_size * num_fields * local_cells \
            + double_buffer_mem \
            + self.value_size * num_pml_fields * local_pml_cells
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
