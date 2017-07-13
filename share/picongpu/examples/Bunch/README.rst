Bunch: Thomson scattering from laser electron-bunch interaction
===============================================================

.. sectionauthor:: Richard Pausch <r.pausch (at) hzdr.de>
.. moduleauthor:: Richard Pausch <r.pausch (at) hzdr.de>, Rene Widera <r.widera (at) hzdr.de>

This is a simulation of an electron bunch that collides head-on with a laser pulse.
Depending on the number of electrons in the bunch, their momentum and their distribution and depending on the laser wavelength and intensity, the emitted radiation differs.
A general description of this simulation can be found in [PauschDipl]_.
A detailed analysis of this bunch simulation can be found in [Pausch13]_.
A theoretical study of the emitted radiation in head-on laser electron collisions can be found in [Esarey93]_.

This test simulates an electron bunch with a relativistic gamma factor of gamma=5.0 and with a laser with a_0=1.0.
The resulting radiation should scale with the number of real electrons (incoherent radiation).

References
----------

.. [PauschDipl]
        Richard Pausch.
        *Electromagnetic Radiation from Relativistic Electrons as Characteristic Signature of their Dynamics*,
        Diploma Thesis TU Dresden (2012),
        http://www.hzdr.de/db/Cms?pOid=38997

.. [Pausch13]
        R. Pausch, A. Debus, R. Widera, K. Steiniger, A. Huebl, H. Burau, M. Bussmann, U. Schramm.
        *How to test and verify radiation diagnostics simulations within particle-in-cell frameworks*,
        Nuclear Instruments and Methods in Physics Research Section A (2013),
        http://dx.doi.org/10.1016/j.nima.2013.10.073

.. [Esarey93]
        E. Esarey, S. Ride, P. Sprangle.
        *Nonlinear Thomson scattering of intense laser pulses from beams and plasmas*,
        Physical Review E (1993),
        http://dx.doi.org/10.1103/PhysRevE.48.3003
