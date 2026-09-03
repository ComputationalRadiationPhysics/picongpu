"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import logging
import traceback
from typing import Callable, Literal

import numpy as np
from picmistandard import PICMI_Extension
from pydantic import ConfigDict, Field, PrivateAttr, computed_field
from sympy import Expr, Symbol, lambdify, symbols

from picongpu.pypicongpu import species
from picongpu.pypicongpu.util import decorating_class

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


@decorating_class("density_function")
class AnalyticDistribution(PICMI_Extension):
    """
    This class represents a plasma with a density defined by an analytic expression.

    The function must be constructed using sympy functions
    to enable code generation and manipulation.
    This is a slight deviation from the PICMI standard.
    Furthermore, we don't implement substitution of variables
    as suggested in the PICMI standard.
    Instead we propose that you write your function
    with further variables as keyword arguments
    and substitute them yourself before handing it over to AnalyticDistribution.
    See the end-to-end tests for examples of this.

    Writing such functions (or rather writing sympy in general)
    comes with a few pitfalls but also advantages as listed below.
    Make sure that you familiarise yourself with writing sympy.

    Advantages:
    - The sympy language is closer to mathematical language than to coding
      which might make it more natural to use for some physicists.
    - You can extract the exact distribution
      that was used from the member `density_expression`
      and use it any way you'd use any sympy expression.
      In particular, you can print it to various formats,
      say, LaTeX for automated inclusion in papers.
    - We can easily evaluate it from within python.
      This is what the __call__ operator does.
      However, code generation here can have some difficulties
      with advanced broadcasting for numpy variables.
      The operator implements a fallback in such cases.
      This fallback might be slightly slower on large inputs.
      Try rewriting your function, in case you experience performance problems.

    Pitfalls:
    - We don't handle vectors yet.
      But that's probably not too important for density expressions.
      Just be explicit handling multiple vector components for now.
    - Some operations might compile to suboptimal code
      concerning numerical performance and stability.
      Experts might want to inspect the generated C++ code
      and/or check the unit tests for the PMAccPrinter
      to find the precise mapping of sympy expressions
      to PMAcc code.
      Please approach us if you should stumble across this.
    - Control flow is an interesting topic in this regard.
      With respect to your three position coordinates
      (which will be sympy symbols internally),
      you must use pure sympy, e.g.,
      replacing if-conditions with sympy.Piecewise and so on.
      With respect to further parameters,
      you're free to use any python construct you want
      (if-conditions, loops, etc.).
    - sympy.Piecewise has the potentially surprising property
      that any pieces that you leave undefined are interpreted as nan.
      This implies that adding two complementary sympy.Piecewise
      renders the whole expression nan and not -- as you might expect --
      defined on the union of the defined regions.
      There are two options to circumvent this:
      Either you can define multiple sympy.Piecewise with
      (0.0, True) as the last condition which means 0 everywhere else.
      (Make sure it's the last!)
      Summing those up, works just as you'd expect.
      Alternatively, you can define only the (expression, condition) tuples
      and assemble them in a sympy.Piecewise in one go.
      That's the way chosen in the end-to-end tests.

    Parameters:
        density_expression (Callable):
            A Python function that takes x, y, z coordinates (in SI units)
            and returns the density (in SI units) at that point.
            It should use sympy functionality.
        directed_velocity (3-tuple of float):
            A collective velocity for the particle distribution.
            (currently untested)
    """

    density_function: Callable[[Symbol, Symbol, Symbol], Expr]
    rms_velocity: Literal[(0.0, 0.0, 0.0)] = (0.0, 0.0, 0.0)
    directed_velocity: list[float] = Field(default_factory=lambda: [0, 0, 0])
    _warned_about_lambdify_failure: bool = PrivateAttr(False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    def density_expression(self) -> Expr:
        x, y, z = symbols("x,y,z")
        return self.density_function(x, y, z) + (0 * x * y * z)

    def get_as_pypicongpu(self, _):
        return species.operation.densityprofile.FreeFormula(density_expression=self.density_expression)

    def picongpu_get_rms_velocity_si(self) -> tuple[float, float, float]:
        return self.rms_velocity

    def get_picongpu_drift(self) -> species.operation.momentum.Drift | None:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if all(v == 0 for v in self.directed_velocity):
            return None

        return species.operation.momentum.Drift.from_velocity(
            self.directed_velocity  # type: ignore[arg-type]
        )

    def __call__(self, *args, **kwargs):
        args = tuple(np.asarray(a) for a in args)
        try:
            # This produces faster code but the code generation is not perfect.
            # There are cases where the generated code can't handle broadcasting properly.
            return lambdify(symbols("x,y,z"), self.density_expression, "numpy")(*args, **kwargs)
        # We explicitly want this to be as broad as possible
        # because we have a second shot.
        # There should be no instances of this being dangerous during idiomatic use of this functionality.
        except Exception:
            if not self._warned_about_lambdify_failure:
                message = (
                    "Sympy did not manage to produce proper numpy code for your AnalyticDistribution. "
                    "If you run into performance problems, try to rewrite your function. "
                    "Here's the original error message:"
                )
                logging.warning(message)
                logging.warning(traceback.format_exc())
                logging.warning("Continuing operation using a slower serialised version now.")
                self._warned_about_lambdify_failure = True
        # This basically calls the original function in a big loop.
        # Slower but more reliable in some cases of difficult broadcasting.
        return np.vectorize(self.density_function)(*args, **kwargs)
