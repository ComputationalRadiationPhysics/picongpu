"""
This file is part of PIConGPU.

Copyright 2017-2023 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+


If more parameters are added here, make sure to adapt the corresponding
*.param files (located in the 'simulation_defines' directory of this example)
that define those values within picongpu by adding the usual
C language macro statements. The names of the macros should be the uppercase
versions of the names provided here with an additional PARAM_ prefix.

In order to visualize parameters via jupyter notebooks and ipython widgets,
they need to be at least of class Parameter (or inherited).
"""

from picongpu.input.parameters import Parameter


dt = 1.39e-16

PARAMETERS = {
    "laser": [
        Parameter(name="_A0", ptype="compile", unit="1", default=1.5, range=(0.1, 50.01)),
        Parameter(
            name="Wave_Length_SI",
            ptype="compile",
            unit="nm",
            default=800.0,
            range=(400.0, 1400.0),
        ),
        Parameter(
            name="Pulse_Duration_SI",
            ptype="compile",
            unit="fs",
            default=5.0,
            range=(1.0, 150.0),
        ),
    ],
    "target": [
        Parameter(
            name="Base_Density_SI",
            ptype="compile",
            unit="1/m^3",
            default=1.0e25,
            range=(1.0e20, 1.0e26),
        ),
    ],
    "resolution": [
        Parameter(
            name="TBG_steps",
            ptype="run",
            unit="ps",
            default=1.0,
            range=(0.1, 10.0),
            pic_to_SI=lambda steps: steps * dt,
            pic_from_SI=lambda time: int(round(time / dt)),
            label="simulation time",
        )
    ],
}
