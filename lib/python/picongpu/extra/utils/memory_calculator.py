#!/usr/bin/env python

"""
# SPDX-FileCopyrightText: Marco Garten, Sergei Bastrakov, Brian Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import numpy as np
import numpy.typing as nptype

import pydantic
from pydantic import ConfigDict
import typeguard


class MemoryCalculator(pydantic.BaseModel):
    """
    Memory requirement calculation tool for PIConGPU

    Contains calculation for fields, particles, random number generator and the calorimeter plugin.

    In-situ methods other than the caloriometer so far use up negligible amounts of memory on the device.
    """

    simulation_dimension: int

    """
    numerical order of the assignment function of the chosen particle shape

    CIC : order 1
    TSC : order 2
    PQS : order 3
    PCS : order 4
    (see ``species.param``)
    """
    particle_shape_order: int = 2

    # pml border size, in cells, see ``fieldAbsorber.param``:``NUM_CELLS``
    pml_border_size: nptype.NDArray = np.array(((12, 12), (12, 12), (12, 12)))

    # precision used by PIConGPU, see ``precision.param``
    precision: int = 32

    # size of super cell in cells, see ``memory.param``
    super_cell_size: nptype.NDArray = np.array((8, 8, 4))

    # in super cells, see ``memory.param``
    guard_size: nptype.NDArray = np.array((1, 1, 1))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **keyword_arguments):
        pydantic.BaseModel.__init__(self, **keyword_arguments)
        self._check_dimensions_of_arrays()
        self._shrink_to_simulation_dimension()
        self._check()

    @staticmethod
    def get_value_size(precision: int) -> int:
        """
        get size of basic value depending on picongpu precision

        @return unit: bytes
        """

        if precision == 32:
            # bytes
            return np.float32().itemsize
        elif precision == 64:
            # bytes
            return np.float64().itemsize
        else:
            raise ValueError("unsupported precision {precision}")

    @staticmethod
    def get_predefined_attribute_dict(simulation_dimension: int, precision: int) -> dict[str, int]:
        """get dictionary describing the size of each predefined attribute in bytes"""

        size_float_X = MemoryCalculator.get_value_size(precision)
        size_uint8 = np.uint8().itemsize
        size_uint32 = np.uint32().itemsize
        # we assume 1 byte per bool
        size_bool = 1

        return {
            "momentum": 3 * size_float_X,
            "position": simulation_dimension * size_float_X,
            "momentumPrev1": 3 * size_float_X,
            "weighting": size_float_X,
            "particleId": np.uint64().itemsize,
            "weightingDampningFactor": size_float_X,
            "probeE": 3 * size_float_X,
            "probeB": 3 * size_float_X,
            "radiationMask": size_bool,
            "transitionRadiationMask": size_bool,
            "boundElectrons": size_float_X,
            "atomicStateCollectionIndex": size_uint32,
            "processClass": size_uint8,
            "transitionIndex": size_uint32,
            "binIndex": size_uint32,
            "accepted": size_bool,
            "atomicPhysicsIonParticleAttributes": size_float_X + 3 * size_uint32 + size_uint8 + size_bool,
            "totalCellIdx": simulation_dimension * size_uint32,
        }

    @typeguard.typechecked
    def _check_cell_extent(self, cell_extent: nptype.NDArray):
        """check cell extent is consistent with configuration"""
        if (cell_extent).ndim != 1:
            raise ValueError("cell_extent must be 1D array")

        if (cell_extent).shape[0] != self.simulation_dimension:
            raise ValueError("simulation_dimension and dimension of cell_extent must match.")
        if not (np.all(cell_extent > 0)):
            raise ValueError("number cells must be > 0 in all dimensions")

        if np.any(cell_extent % self.super_cell_size != 0):
            raise ValueError(
                "device number cells must be an integer multiple of the super_cell_size,"
                " please set super_cell_size to a correct value"
            )

    def _check_dimensions_of_arrays(self):
        """check all set array have expected dimension"""
        if (self.pml_border_size).ndim != 2:
            raise ValueError("pml_border_size must be 2D array")
        if (self.super_cell_size).ndim != 1:
            raise ValueError("super_cell_size must be 1D array")
        if (self.guard_size).ndim != 1:
            raise ValueError("guard_size must be 1D array")

    def _shrink_to_simulation_dimension(self):
        if self.simulation_dimension == 2:
            if (self.pml_border_size).shape[0] == 3:
                self.pml_border_size = self.pml_border_size[:2]

            if (self.super_cell_size).shape[0] == 3:
                self.super_cell_size = self.super_cell_size[:2]

            if (self.guard_size).shape[0] == 3:
                self.guard_size = self.guard_size[:2]

    @typeguard.typechecked
    def _check(self):
        """check configuration is sensible"""
        if self.simulation_dimension > 3 or self.simulation_dimension < 2:
            raise ValueError("PIConGPU only supports 2D or 3D simulations.")

        if (self.precision != 32) and (self.precision != 64):
            raise ValueError("PIConGPU only supports either 32 or 64 bits precision.")

        if not (np.all(self.super_cell_size > 0)):
            raise ValueError("super_cell_size must be > 0 in all dimensions")

    @typeguard.typechecked
    def memory_required_by_cell_fields(
        self, cell_extent: nptype.NDArray, number_of_temporary_field_slots: int = 1
    ) -> int:
        """
        Memory required for cell fields on a specific device(GPU/CPU/...)

        @attention In PIConGPU different devices may handle different number of cells. This naturally also changes the
            memory required on each device. This function returns the memory required by cell fields on one single
            device handling the specified cell extent, not the global memory requirement!
        @attention this method requires the actual extent of cells on device, not just the number of cells!

        @param cell_extent device cell extent
        @param number_of_temporary_field_slots number of temporary field slots, see ``memory.param``

        @return unit: bytes
        """
        self._check_cell_extent(cell_extent)

        # PML size cannot exceed the local grid size
        pml_border_size = np.minimum(self.pml_border_size, cell_extent[:, np.newaxis])

        # one scalar each temporary field and E, B, J field component
        number_fields = 3 * 3 + number_of_temporary_field_slots

        # number of additional PML field components: when enabled,
        # 2 additional scalar fields for each of E and B field component
        number_pml_fields = 2 * 2 * 3

        number_local_cells = int(np.prod(cell_extent + self.super_cell_size * 2 * self.guard_size))
        number_pml_cells = int(np.prod(cell_extent) - np.prod(cell_extent - np.sum(pml_border_size, axis=1)))
        number_double_buffer_cells = int(np.prod(cell_extent + self.particle_shape_order) - np.prod(cell_extent))

        size_float_X = MemoryCalculator.get_value_size(self.precision)

        double_buffer_memory = number_double_buffer_cells * number_fields * size_float_X
        cell_memory = number_local_cells * number_fields * size_float_X
        pml_memory = number_pml_cells * number_pml_fields * size_float_X

        return cell_memory + double_buffer_memory + pml_memory

    @typeguard.typechecked
    def memory_required_by_super_cell_fields(
        self,
        super_cell_extent: nptype.NDArray,
        number_atomic_states_by_atomic_physics_ion_species: list[int],
        number_electron_histogram_bins: int,
        ipd_active: bool = True,
    ) -> int:
        """
        Memory required for super cell fields on a specific device(GPU/CPU/...)

        @attention In PIConGPU different devices may handle different number of super cells. This naturally also changes
            the memory required on each device. This function returns the memory required by cell fields on one single
            device handling the specified cell extent, not the global memory requirement!

        @param super_cell_extent device super_cell extent
        @param number_atomic_states_by_atomic_physics_ion_species number of atomic states of each atomic physics ion
            species
        @param number_electron_histogram_bins number of bins in the AtomicPhysics(FLYonPIC) electron histograms
        @param IPDactive is IPD active, see ``IPDModel.param``

        @return unit: bytes
        """
        self._check_cell_extent(super_cell_extent * self.super_cell_size)

        number_cells_per_supercell = int(np.prod(self.super_cell_size))
        # bytes
        size_float_X = MemoryCalculator.get_value_size(self.precision)
        # bytes
        size_uint32 = np.uint32().itemsize

        number_process_class_groups_in_rate_cache = 5

        # bytes
        size_rate_caches = 0
        for number_states in number_atomic_states_by_atomic_physics_ion_species:
            size_rate_caches += number_states * (size_float_X * number_process_class_groups_in_rate_cache + size_uint32)

        size_rejection_probability_cache_cell = size_float_X * number_cells_per_supercell
        size_rejection_probability_cache_bin = size_float_X * number_electron_histogram_bins
        size_field_energy_use_cache = size_float_X * number_cells_per_supercell

        size_electron_histogram = 3 * size_float_X * number_electron_histogram_bins + size_float_X
        size_shared_ressources_over_subscribed = size_uint32
        size_shared_found_unbound = size_uint32
        size_time_remaining = size_float_X
        size_time_step = size_float_X

        per_super_cell_memory = (
            size_rate_caches
            + size_rejection_probability_cache_cell
            + size_rejection_probability_cache_bin
            + size_field_energy_use_cache
            + size_electron_histogram
            + size_shared_ressources_over_subscribed
            + size_shared_found_unbound
            + size_time_remaining
            + size_time_step
        )

        ipd_sum_weight_all = size_float_X
        ipd_sum_weight_electrons = size_float_X
        ipd_sum_temperature_functional = size_float_X
        ipd_sum_charge_number_ions = size_float_X
        ipd_sum_charge_number_ions_squared = size_float_X
        ipd_debye_length = size_float_X
        ipd_zstar = size_float_X
        ipd_temperature_energy = size_float_X

        if ipd_active:
            per_super_cell_memory += (
                ipd_sum_weight_all
                + ipd_sum_weight_electrons
                + ipd_sum_temperature_functional
                + ipd_sum_charge_number_ions
                + ipd_sum_charge_number_ions_squared
                + ipd_debye_length
                + ipd_zstar
                + ipd_temperature_energy
            )

        number_local_super_cells = int(np.prod(super_cell_extent))

        return number_local_super_cells * per_super_cell_memory

    @typeguard.typechecked
    def memory_required_by_particles_of_species(
        self,
        particle_filled_cells: nptype.NDArray,
        particles_per_cell: int,
        species_attribute_list: list[str],
        custom_attributes_size_dict: dict[str, int] = {},
    ) -> int:
        """
        Memory required for a species' particles per device.

        @detail species flag/constants do not occupy global device memory, since they are compiled in.
        @attention use particles_per_cell=1 if passing the number of particles directly!

        @param particle_filled_cells either number of particle filled cells, extent of particle filled cells,
            or number of macro-particles of species on device
        @param particles_per_cell number of particles of the species per cell
        @param species_attribute_list list of species attribute names of species, see ``species.param``,
            for example ["momentum", "position", "weighting"]
        @param additional_attributes list of size of additional attributes, in bytes, for example {"custom":4}

        @return unit: bytes
        """

        # in byte
        minimum_particle_attributes = {"multiMask": 1, "cellIndex": 2}

        predefined_attribute_size_dict = MemoryCalculator.get_predefined_attribute_dict(
            self.simulation_dimension, self.precision
        )

        # cells filled by the target species
        number_particle_cells = int(np.prod(particle_filled_cells))

        # byte
        mem_per_particle = 0

        for attribute in species_attribute_list:
            if attribute in predefined_attribute_size_dict:
                # bit
                mem_per_particle += predefined_attribute_size_dict[attribute]
            elif attribute in custom_attributes_size_dict:
                # bit
                mem_per_particle += custom_attributes_size_dict[attribute]
            else:
                raise ValueError(
                    "size of species attribute {attribute} unknown, not a known predefined attribute and"
                    "not specified by user"
                )

        for attribute in minimum_particle_attributes.keys():
            # byte
            mem_per_particle += minimum_particle_attributes[attribute]

        # byte
        req_mem = int(np.ceil(number_particle_cells * mem_per_particle * particles_per_cell))

        return req_mem

    @typeguard.typechecked
    def memory_required_by_random_number_generator(
        self, cell_extent: nptype.NDArray, generator_method: str = "XorMin"
    ) -> int:
        """
        Memory reserved for the random number generator state on device per device.

        See ``random.param`` for a choice of random number generators.

        If you find that your required RNG state is large (> 300 MB) please adjust the ``reservedGpuMemorySize``
        in ``memory.param`` to fit the RNG state and some.

        @param cell_extent cell extent per device
        @param generator_method random number generator method - influences the state size per cell
            possible options: "XorMin", "MRG32k3aMin", "AlpakaRand"
            - (GPU default: "XorMin")
            - (CPU default: "AlpakaRand")

        @return unit: bytes
        """
        self._check_cell_extent(cell_extent)

        if generator_method == "XorMin":
            # bytes
            state_size_per_cell = 6 * 4
        elif generator_method == "MRG32k3aMin":
            # bytes
            state_size_per_cell = 6 * 8
        elif generator_method == "AlpakaRand":
            # bytes
            state_size_per_cell = 7 * 4
        else:
            raise ValueError(
                f"{generator_method} is not a known RNG for PIConGPU. Please choose one of the following: "
                "'XorMin', 'MRG32k3aMin', 'AlpakaRand'"
            )

        # CORE + BORDER region of the device, GUARD currently has no RNG state
        number_local_cells = int(np.prod(cell_extent))

        req_mem = state_size_per_cell * number_local_cells
        return req_mem

    @typeguard.typechecked
    def memory_required_by_calorimeter(self, number_energy_bins: int, number_yaw_bins: int, number_pitch_bins: int):
        """
        Memory required by the particle calorimeter plugin.

        Each of the (``n_energy`` x ``n_yaw`` x ``n_pitch``) bins requires a value (32/64 bits).
        The whole calorimeter is represented twice on each device, once for particles in the simulation and once
        for the particles that leave the box.

        @param number_energy_bins number of bins on the energy axis
        @param number_yaw_bins number of bins for the yaw angle
        @param number_pitch_bins number of bins for the pitch angle

        @return unit: bytes
        """
        # bytes
        req_mem_per_bin = self.get_value_size(self.precision)

        total_number_bins = number_energy_bins * number_yaw_bins * number_pitch_bins

        # one calorimeter instance for particles in the box
        # another calorimeter instance for particles leaving the box
        # makes a factor of 2 for the required memory
        req_mem = req_mem_per_bin * total_number_bins * 2

        return req_mem
