Emission of Bremsstrahlung from Laser-foil interaction
======================================================

* author:      Heiko Burau <h.burau (at) hzdr.de>
* maintainer:  Heiko Burau <h.burau (at) hzdr.de>

This is a simulation of a flat solid density target hit head-on by a high-intensity laser pulse. 
At the front surface free electrons are accelerated up to ultra relativistic energies and start travelling through the bulk then. 
Meanwhile, due to ion interaction, the hot electrons lose a small fraction of their kinectic energy in favor of emission of Bremsstrahlung-photons. 
Passing over the back surface hot electrons are eventually reflected and re-enter the foil in opposite direction. 
Because of the ultra-relativistic energy Bremsstrahlung (BS) is continuously emitted mainly along the direction of motion of the electron.
The BS-module models the electron-ion scattering as three single processes, including electron deflection, electron deceleration and photon creation with respect to the emission angle. 
Details of the implementation and the numerical model can be found in [BurauDipl]_. 
Details of the theoretical description can be found in [Jackson]_ and [Salvat]_.

This 2D test simulates a laser pulse of a_0=40, lambda=0.8µm, w0=1.5µm in head-on collision with a fully pre-ionized gold foil of 2µm thickness.

Checks
------

 - check appearence of photons moving along (forward) and against (backward) the incident laser pulse direction.
 - check photon energy spectrum in both directions for the forward moving photons having a higher energy.

References
----------

.. [BurauDipl]
    Heiko Burau.
    Entwicklung und Überprüfung eines Photonenmodells für die Abstrahlung durch hochenergetische Elektronen,
    Diploma Thesis TU Dresden (2016),
    https://doi.org/10.5281/zenodo.192116

.. [Jackson]
    Jackson, J. David.
    Electrodynamics,
    Wiley‐VCH Verlag GmbH & Co. KGaA (1975),
    http://onlinelibrary.wiley.com/doi/10.1002/9783527600441.oe014

.. [Salvat]
    F. Salvat, J. Fernández-Varea, J. Sempau, and X. Llovet.
    Monte carlo simulation of bremsstrahlung emission by electrons,
    Radiation Physics and Chemistry (2006),
    http://dx.doi.org/10.1016/j.radphyschem.2005.05.008

