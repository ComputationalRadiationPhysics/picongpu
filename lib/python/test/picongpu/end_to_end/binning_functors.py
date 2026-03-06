"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import partial
import numpy as np
import sympy
from picongpu.picmi.diagnostics.binning import (
    Binning,
    BinningAxis,
    BinningFunctor,
    BinSpec,
)
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from scipy.constants import speed_of_light

from .arbitrary_parameters import (
    ALL_ORIGINS_WITHOUT_GUARDS,
    ALL_PRECISIONS,
    ALL_UNITS,
    CELL_SIZE,
    EPSILON,
    NUMBER_OF_CELLS,
    NUMBER_OF_GUARD_CELLS,
)


def particle_density(particle):
    return particle.get("weighting")


def origin_no_guards_check(particle):
    positions = {
        (precision, unit): {
            origin: np.array(particle.get("position", origin=origin, precision=precision, unit=unit))
            for origin in ALL_ORIGINS_WITHOUT_GUARDS
            if origin != "moving_window" or precision == "sub_cell"
        }
        for precision in ALL_PRECISIONS
        for unit in ALL_UNITS
    }

    all_combinations = tuple(
        x for ps in positions.values() for a in ps.values() for b in ps.values() for x in zip(a, b)
    )
    return sum(sympy.Piecewise((1, sympy.Eq(*x)), (0, True)) for x in all_combinations) / len(all_combinations)


def _subtract_guards(position, unit, timestep):
    if unit == "cell":
        return position - NUMBER_OF_GUARD_CELLS

    if unit == "si":
        return position - NUMBER_OF_GUARD_CELLS * CELL_SIZE

    if unit == "pic":
        return position - NUMBER_OF_GUARD_CELLS * CELL_SIZE / (timestep * speed_of_light)


def origin_with_guards_check(particle, timestep):
    positions = {
        (precision, unit): {
            origin: np.array(particle.get("position", origin=origin, precision=precision, unit=unit))
            for origin in ["local_with_guards", "local"]
            if origin != "moving_window" or precision == "sub_cell"
        }
        for precision in ALL_PRECISIONS
        for unit in ALL_UNITS
    }

    all_combinations = sum(
        (
            tuple(zip(p["local"], _subtract_guards(p["local_with_guards"], unit, timestep)))
            for (_, unit), p in positions.items()
        ),
        tuple(),
    )
    return sum(sympy.Piecewise((1, sympy.Eq(*x)), (0, True)) for x in all_combinations) / len(all_combinations)


def unit_no_guards_check(particle, timestep):
    positions = {
        (precision, origin): {
            unit: particle.get("position", origin=origin, precision=precision, unit=unit) for unit in ALL_UNITS
        }
        for precision in ALL_PRECISIONS
        for origin in ALL_ORIGINS_WITHOUT_GUARDS
        if origin != "moving_window" or precision == "sub_cell"
    }
    return sum(
        sympy.Piecewise(
            (1, sympy.Abs(ps["cell"][i] * CELL_SIZE[i] - ps["si"][i]) < EPSILON),
            (0, True),
        )
        + sympy.Piecewise(
            (
                1,
                sympy.Abs(ps["pic"][i] * speed_of_light * timestep - ps["si"][i]) < EPSILON,
            ),
            (0, True),
        )
        for ps in positions.values()
        for i in range(len(NUMBER_OF_CELLS))
    ) / (2 * len(NUMBER_OF_CELLS) * len(positions))


def position(particle, i, origin, precision, unit):
    return particle.get("position", origin=origin, precision=precision, unit=unit)[i]


def position_binning_for(species, timestep):
    # Sorry, this way of testing is not really great to debug.
    # The deposition_functors we'll use perform a number of checks at once.
    # They'll count the number of passed checks and normalise it at the end.
    # So, they'll return 1 for each particle, i.e., the number of particles after accumulation.
    # This is awful to debug, I'm aware of that.
    # But unfortunately, the other ways I could come up with were quite intensive in their resource usage.
    # The best I could come up with for debugging is cd'ing into the setup directory
    # and manually changing the generated C++ files.
    # That worked quite okay.
    kwargs = dict(
        axes=[
            BinningAxis(
                functor=BinningFunctor(name="dummy", functor=lambda x: 0, return_type=float),
                bin_spec=BinSpec("linear", -0.5, 0.5, 1),
            )
        ],
        species=species,
        period=TimeStepSpec[:],
        openPMD={"hdf5": {"dataset": {"chunks": "auto"}}},
        openPMDExt="h5",
    )
    return [
        Binning(
            name=f"origin_{species.name}",
            deposition_functor=BinningFunctor(
                name="origin_check",
                functor=partial(origin_no_guards_check),
                return_type="double",
            ),
            **kwargs,
        ),
        Binning(
            name=f"unit_{species.name}",
            deposition_functor=BinningFunctor(
                name="unit_check",
                functor=partial(unit_no_guards_check, timestep=timestep),
                return_type="double",
            ),
            **kwargs,
        ),
        Binning(
            name=f"origin_guards_{species.name}",
            deposition_functor=BinningFunctor(
                name="origin_with_guards_check",
                functor=partial(origin_with_guards_check, timestep=timestep),
                return_type="double",
            ),
            **kwargs,
        ),
    ]


POSITION_AXES = [
    BinningAxis(
        BinningFunctor(
            # We prefer `partial` over lambda functions in this situation
            # because of lambda's late binding.
            name=f"position{i}",
            functor=partial(position, i=i, origin="total", precision="cell", unit="cell"),
            return_type="double",
        ),
        BinSpec("linear", 0, NUMBER_OF_CELLS[i], NUMBER_OF_CELLS[i]),
        use_overflow_bins=False,
    )
    for i in range(3)
]


def density_binning_for(species):
    return [
        Binning(
            name=f"particleDensity_{species.name}",
            deposition_functor=BinningFunctor(
                name=f"particle_density_{species.name}",
                functor=particle_density,
                return_type=float,
            ),
            # Reversing this list is more convenient for comparison with openPMD data.
            axes=POSITION_AXES[::-1],
            species=species,
        )
    ]


def binning_diagnostics(all_species, timestep):
    return sum((density_binning_for(species) + position_binning_for(species, timestep) for species in all_species), [])
