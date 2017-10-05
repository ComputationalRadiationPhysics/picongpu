from eupraxia_picongpu.parameters import *

# if more params are added, make sure to add their names and sensible default
# values to uiParamDefines.param within picongpu simulation_defines and adapt
# the *.param files that define those values
# in order to visualize parameters they need to be at least of type UiParameter or inherited

PARAMETER_LIST = [
    LogScaledParameter(name="Density", ptype="compile", unit="1/m^3",
                       default=25, slider_min=20, slider_max=26, slider_step=1, base=10),

    UiParameter(name="Laser_A0", ptype="compile",
                unit="", default=1.5, slider_min=0.1, slider_max=50.01, slider_step=0.1),

    LinearScaledParameter(name="Laser_Wavelength", ptype="compile", unit="m", default=0.8,
                          slider_min=0.4, slider_max=1400.1, slider_step=0.1, scale_factor=1.e-6),

    LinearScaledParameter(name="LASER_PULSE_LENGTH", ptype="compile", unit="s", default=5,
                          slider_min=1, slider_max=150, slider_step=1, scale_factor=1.e-15),
]
