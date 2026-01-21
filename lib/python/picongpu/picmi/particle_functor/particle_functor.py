"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Any, Callable, Iterable

from sympy import Expr, Symbol, symbols
from typeguard import typechecked

from picongpu.picmi.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.particle_functor import (
    ParticleFunctor as PyPIConGPUParticleFunctor,
    UnitDimension as PyPIConGPUUnitDimension,
    generate_preamble,
)

_COORDINATE_SYSTEM = {
    (
        origin.lower(),
        precision.lower(),
        unit.lower(),
    ): tuple(Symbol(f"{c}_{precision.lower()}_{unit.lower()}") for c in coords)
    for (origin, coords) in (
        ("TOTAL", ("xt", "yt", "zt")),
        ("GLOBAL", ("xg", "yg", "zg")),
        ("LOCAL", ("xl", "yl", "zl")),
        ("MOVING_WINDOW", ("xmw", "ymw", "zmw")),
        ("LOCAL_WITH_GUARDS", ("xlg", "ylg", "zlg")),
        ("CELL", ("xc", "yc", "zc")),
    )
    for precision in ("CELL", "SUB_CELL")
    for unit in ("CELL", "PIC", "SI")
}


class Particle:
    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        NotImplementedError()

    def finalize(self, expression, name=None, return_type=None):
        return expression


@typechecked
class AbstractParticle(Particle):
    needs_total_position = False

    def __init__(self):
        self.used_attributes = {}

    def get_attribute_map(self):
        return self.used_attributes

    def get(self, attribute, **kwargs) -> Expr | Iterable[Expr]:
        if attribute == "position":
            origin = kwargs.get("origin", "total")
            precision = kwargs.get("precision", "cell")
            unit = kwargs.get("unit", "cell")
            self.needs_total_position = self.needs_total_position or (origin.lower() not in ["cell", "local"])
            my_symbols = _COORDINATE_SYSTEM[(origin, precision, unit)]
            self.used_attributes |= {my_symbols: ("position", origin, precision, unit)}

        elif attribute == "momentum":
            my_symbols = symbols("px,py,pz")
            self.used_attributes |= {my_symbols: "momentum"}

        elif attribute == "momentumPrev1":
            my_symbols = symbols("p1x,p1y,p1z")
            self.used_attributes |= {my_symbols: "momentumPrev1"}

        elif attribute in ["gamma", "kinetic energy", "velocity"]:
            # This relies on python dictionaries having a stable ordering.
            # We first add mass and momentum
            # and later use their symbols inside of the same preamble.
            self.get("mass")
            self.get("momentum")
            if attribute == "gamma":
                my_symbols = Symbol("gamma")
            elif attribute == "kinetic energy":
                my_symbols = Symbol("Ekin")
            elif attribute == "velocity":
                my_symbols = symbols("vx,vy,vz")
            else:
                raise ValueError("Reached impossible path.")
            self.used_attributes |= {my_symbols: attribute}

        else:
            my_symbols = Symbol(attribute)
            self.used_attributes |= {my_symbols: attribute}

        return my_symbols

    def finalize(self, expression, name=None, return_type=None):
        return expression


@typechecked
class ParticleFunctor:
    def __init__(
        self,
        name: str,
        functor: Callable[[Particle], Any],
        return_type: type | str = float,
        unit_dimension: UnitDimension | None = None,
    ):
        self.name = name
        self.functor = functor
        self.return_type = return_type
        self.unit_dimension = unit_dimension or UnitDimension()

    def get_as_pypicongpu(self, mode) -> PyPIConGPUParticleFunctor:
        particle = AbstractParticle()
        functor_expression = self(particle)
        return PyPIConGPUParticleFunctor(
            name=self.name,
            functor_expression=functor_expression,
            functor_preamble=generate_preamble(particle.get_attribute_map(), mode=mode),
            return_type=self.return_type,
            unit_dimension=PyPIConGPUUnitDimension(unit_dimension=self.unit_dimension.unit_vector.tolist()),
            needs_total_position=particle.needs_total_position,
        )

    def __call__(self, particle):
        expression = self.functor(particle)
        return particle.finalize(expression, self.name, self.return_type)
