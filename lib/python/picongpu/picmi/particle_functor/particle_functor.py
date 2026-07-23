"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from inspect import signature
from typing import Any, Callable, Iterable

from pydantic import BaseModel, model_validator
from pydantic._internal._model_construction import ModelMetaclass
from sympy import Expr, Symbol, symbols

from picongpu.picmi.particle_functor.rng_arg import RNGArg
from picongpu.picmi.particle_functor.unit_dimension import UnitDimension
from picongpu.pypicongpu.particle_functor import (
    ParticleFunctor as PyPIConGPUParticleFunctor,
    UnitDimension as PyPIConGPUUnitDimension,
    generate_preamble,
)
from picongpu.pypicongpu.util import alt

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


class _DecoratingMeta(ModelMetaclass):
    """
    Metaclass enabling @ParticleFunctor and @ParticleFunctor(kwargs) decorator syntax.

    decorating_class cannot be used with pydantic BaseModel because pydantic's
    __init__ only accepts **data as keyword arguments, while decorating_class
    passes the decorated callable as a class-level positional argument.
    """

    def __call__(cls, *args, **kwargs):
        sig = signature(cls)
        first_param_name = next(iter(sig.parameters.values())).name

        # Case 1: @ParticleFunctor (callable as first positional arg, no kwargs)
        if args and callable(args[0]) and first_param_name not in kwargs:
            return super().__call__(**{first_param_name: args[0]}, **kwargs)

        # Case 2: @ParticleFunctor(unit_dimension=...) (kwargs only, no callable)
        # Returns a decorator that accepts the decorated function
        if not args and first_param_name not in kwargs:
            decorator_kwargs = dict(kwargs)

            def decorator(func):
                decorator_kwargs[first_param_name] = func
                return super(_DecoratingMeta, cls).__call__(**decorator_kwargs)

            return decorator

        return super().__call__(*args, **kwargs)


class ParticleFunctor(BaseModel, metaclass=_DecoratingMeta):
    """
    A functor that operates on a Particle and returns a sympy expression.

    Usage as decorator::

        @ParticleFunctor
        def kinetic_energy(particle):
            return particle.get("kinetic energy")

        @ParticleFunctor(unit_dimension=M * L / T)
        def momentum_x(particle):
            return particle.get("momentum")[0]

    Or as constructor::

        pf = ParticleFunctor(functor=lambda p: p.get("weighting"), name="density")
    """

    model_config = {"arbitrary_types_allowed": True}

    functor: Callable[[Any], Any]
    name: str | None = None
    return_type: type | str | None = None
    unit_dimension: UnitDimension | None = None
    rng_class: Callable[[Any], Any] = lambda: None

    @model_validator(mode="after")
    def _init(self):
        sig = signature(self.functor)
        if self.name is None:
            self.name = self.functor.__name__
        if self.return_type is None:
            self.return_type = float if sig.return_annotation == sig.empty else sig.return_annotation
        if self.unit_dimension is None:
            self.unit_dimension = UnitDimension()
        rng_classes = [
            cls
            for p in sig.parameters.values()
            if isinstance(p.annotation, type) and issubclass((cls := p.annotation), RNGArg)
        ]
        if len(rng_classes) > 1:
            raise ValueError(
                f"ParticleFunctor can take at most one RNG. You have requested {rng_classes=} in your signature."
            )
        self.rng_class = alt(lambda: rng_classes[0], None) or (lambda: None)
        return self

    def get_as_pypicongpu(self, mode) -> PyPIConGPUParticleFunctor:
        particle = AbstractParticle()
        rng = self.rng_class()
        functor_expression = self(particle) if rng is None else self(particle, rng)
        return PyPIConGPUParticleFunctor(
            name=self.name,
            functor_expression=functor_expression,
            functor_preamble=generate_preamble(
                particle.get_attribute_map() | alt(lambda: rng.get_attribute_map(), {}), mode=mode
            ),
            return_type=self.return_type,
            unit_dimension=PyPIConGPUUnitDimension(unit_dimension=self.unit_dimension.unit_vector.tolist()),
            needs_total_position=particle.needs_total_position,
            rng_info=alt(lambda: rng.model_dump(mode="python"), None),
        )

    def __call__(self, *args):
        return self.functor(*args)


class AbstractParticle(Particle):
    """
    Particle implementation that tracks attribute access and returns sympy symbols.
    """

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