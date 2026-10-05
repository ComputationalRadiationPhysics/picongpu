"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import picmistandard

from picongpu.picmi.species import Species

# Wire the picmi-standard MultiSpecies factory up to our own Species class so that
# the member species it creates are full PIConGPU picmi species (carrying the
# PIConGPU-specific requirements, shapes, pushers, ...).
picmistandard.PICMI_MultiSpecies.Species_class = Species


class MultiSpecies(picmistandard.PICMI_MultiSpecies):
    """
    Multiple species that are initialised from one common initial distribution.

    PICMI-standard semantics (as implemented here): species are initialised
    **independently** by default; a ``MultiSpecies`` is the **explicit** mechanism
    to request **collective** (coordinated) initialisation. All member species share
    the same ``initial_distribution``, so on the C++ level they are placed with a
    single density operation (one ``CreateDensity``) and the remaining members are
    derived from the first one -- yielding exactly the same in-cell positions and
    hence a charge-neutral set-up by construction, irrespective of per-species
    momentum/temperature (which is applied afterwards, per species).

    Each member carries ``density_scale`` equal to its ``proportion``, which maps to
    the species' ``DensityRatio`` and is respected when deriving the members'
    weightings.

    The members are plain :class:`picmi.Species`. Following the PICMI standard,
    pass the ``MultiSpecies`` as a whole to your :class:`picmi.Simulation`,
    either via the declarative constructor
    (``Simulation(..., species=[multi], layouts=[layout])``) or via
    :meth:`picmi.Simulation.add_species`
    (``sim.add_species(species=multi, layout=layout)``); both take a single
    layout for the entire group. Its members can be addressed by index or name
    for further use (e.g. in interactions).

    Grouping is structural: the :class:`picmi.Simulation` stores the
    ``MultiSpecies`` object as one entry, and translation maps that whole entry
    onto a single density operation. There is no per-member marker, so grouping
    is preserved by construction and does not depend on any private state.
    """

    def __iter__(self):
        return iter(self.species_instances_list)
