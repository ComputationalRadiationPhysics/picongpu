"""
This file is part of PIConGPU.

Copyright 2017 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+


If more parameters are added here, make sure to adapt the corresponding
*.param files (located in the 'simulation_defines' directory of this example)
that define those values within picongpu by adding the usual
C language macro statements. The names of the macros should be the uppercase
versions of the names provided here with an additional PARAM_ prefix.

In order to visualize parameters via jupyter notebooks and ipython widgets,
they need to be at least of class UiParameter (or inherited).
"""

from picongpu.input.parameters import LogScaledParameter, LinearScaledParameter

PARAMETER_LIST = [
    LogScaledParameter(
        name="Base_Density_SI", ptype="compile", unit="1/m^3",
        default=25, slider_min=20, slider_max=26,
        slider_step=1, base=10),

    LinearScaledParameter(
        name="_A0", ptype="compile", unit="",
        default=1.5, slider_min=0.1, slider_max=50.01,
        slider_step=0.1),

    LinearScaledParameter(
        name="Wave_Length_SI", ptype="compile", unit="m",
        default=0.8, slider_min=0.4, slider_max=1400.1,
        slider_step=0.1, scale_factor=1.e-6),

    LinearScaledParameter(
        name="Pulse_Length_SI", ptype="compile", unit="s",
        default=5, slider_min=1, slider_max=150,
        slider_step=1, scale_factor=1.e-15),
]
