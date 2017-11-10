.. _model-collisionalIonization:

Collisional Ionization
======================

LTE Models
----------

.. moduleauthor:: Marco Garten

Implemented LTE Model: Thomas-Fermi Ionization according to [More1985]_

Get started here https://github.com/ComputationalRadiationPhysics/picongpu/wiki/Ionization-in-PIConGPU

The implementation of the Thomas-Fermi model takes the following input quantities.

- ion proton number :math:`Z`
- ion species mass density :math:`\rho`
- electron "temperature" :math:`T`

Due to the nature of our simulated setups it is also used in non-equilibrium situations.
We therefore implemented additional conditions to mediate unphysical behavior but introduce arbitrariness.

1. Super-thermal electron cutoff

    We calculate the temperature according to :math:`T_\mathrm{e} = \frac{2}{3} E_\mathrm{kin, e}` in units of electron volts.
    We thereby assume an *ideal electron gas*.
    Via the variable ``CUTOFF_MAX_ENERGY_KEV`` in ``ionizer.param`` the user can exclude electrons with kinetic energy above this value from average energy calculation.
    That is motivated by a lower interaction cross section of particles with high relative velocities.

2. Low ion-density cutoff

    The Thomas-Fermi model displays unphysical behaviour for low ion densities in that it predicts an increasing charge state for decreasing ion densities.
    This occurs already for electron temperatures of 10 eV and the effect increases as the temperature increases.
    For instance in pre-plasmas of solid density targets the charge state would be overestimated where

    - on average electron energies are not large enough for collisional ionization of a respective charge state
    - ion density is not large enough for potential depression
    - electron-ion interaction cross sections are small due to small ion density

    It is strongly suggested to do approximations for **every** setup or material first.
    To that end, a parameter scan with [FLYCHK]_ can help in choosing a reasonable value.

NLTE Models
-----------

.. moduleauthor:: Axel Huebl

in development

.. [More1985]
        R. M. More.
        *Pressure Ionization, Resonances, and the Continuity of Bound and Free States*,
        Advances in Atomic, Molecular and Optical Physics Vol. 21 C, 305-356 (1985),
        https://dx.doi.org/10.1016/S0065-2199(08)60145-1

.. [FLYCHK]
        *FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopic analysis for all elements*,
        H.-K. Chung, M.H. Chen, W.L. Morgan, Yu. Ralchenko, and R.W. Lee,
        *High Energy Density Physics* v.1, p.3 (2005)
        http://nlte.nist.gov/FLY/

