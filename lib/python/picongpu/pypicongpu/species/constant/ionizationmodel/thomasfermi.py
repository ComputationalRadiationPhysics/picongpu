"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .ionizationmodel import IonizationModel


class ThomasFermi(IonizationModel):
    """
    Thomas-Fermi impact ionization

    See table IV from Pressure Ionization, Resonances, and the Continuity of Bound and Free States
      http://www.sciencedirect.com/science/article/pii/S0065219908601451
      doi:10.1016/S0065-2199(08)60145-1

    This ionization model is based on the assumption of an "ion sphere", constructed based on describing electrons as a
    density, a point charge atomic core and a finite atomic potential as a result of matter density.

    In this framework ionization may occur due to due to overlap of adjacent ion spheres lowering the ionization barrier
    and causing electrons to become quasi-free in the system, being bound in resonance states.

    This model is used to calculate an average ionization degree with respect to local charge density and temperature.
    This is extenden to arbitrary temperatures and atoms through fitting parameters and temperature cutoffs.
    """

    PICONGPU_NAME: str = "ThomasFermi"
    """C++ Code type name of ionizer"""
