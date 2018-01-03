"""
This file is part of PIConGPU.

Copyright 2017-2018 PIConGPU contributors
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

pico = 1.e-12
dt_r = 1. / 1.39e-16 * pico

PARAMETERS = {
    'laser': [
        LinearScaledParameter(
            name="_A0", ptype="compile", unit="",
            default=1.5, slider_min=0.1, slider_max=50.01,
            slider_step=0.1),

        LinearScaledParameter(
            name="Wave_Length_SI", ptype="compile", unit="nm",
            default=800.0, slider_min=400.0, slider_max=1400.0,
            slider_step=1, scale_factor=1.e-9),

        LinearScaledParameter(
            name="Pulse_Length_SI", ptype="compile", unit="fs",
            default=5.0, slider_min=1.0, slider_max=150.0,
            slider_step=1, scale_factor=1.e-15),
    ],
    'target': [
        LogScaledParameter(
            name="Base_Density_SI", ptype="compile", unit="1/m^3",
            default=25.0, slider_min=20.0, slider_max=26.0,
            slider_step=1, base=10),
    ],
    'resolution': [
        LinearScaledParameter(
            name="TBG_steps", ptype="run", unit="ps",
            default=1.0, slider_min=0.1, slider_max=10.0,
            slider_step=0.1, scale_factor=dt_r, dtype=int,
            label="simulation time")

    ]
}
