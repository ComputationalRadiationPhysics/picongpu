WarmCopper: Average Charge State Evolution of Copper Irradiated by a Laser
==========================================================================

* author:      Axel Huebl <a.huebl (at) hzdr.de>, Hyun-Kyung Chung
* maintainer:  Axel Huebl <a.huebl (at) hzdr.de>

This setup initializes a homogenous, non-moving, copper block irradiated by a laser with 10^18 W/cm^3 as a benchmark for [SCFLY]_ [#FLYlite]_ atomic population dynamics.
We follow the setup from [FLYCHK]_ page 10, figure 4 assuming a quasi 0D setup with homogenous density of a 1+ ionized copper target.
The laser (not modeled) already generated a thermal electron density at 10, 100 or 1000 eV and a delta-distribution like "hot" electron distribution with 200 keV (directed stream).
The observable of interest is <Z> over time of the copper ions.
For low thermal energies, collisional excitation, de-excitation and recombinations should be sufficient to reach the LTE state after about 0.1-1 ps.
For higher initial temperatures, radiative rates get more relevant and the Non-LTE steady-state solution can only be reached correctly when also adding radiative rates.

.. [#FLYlite] In PIConGPU, we generally refer to the implemented subset of *SCFLY* (solving Non-LTE population kinetics) as *FLYlite*.

References
----------

.. [FLYCHK]
        H.-K. Chung, M.H. Chen, W.L. Morgan, Y. Ralchenko, R.W. Lee.
        *FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopic analysis for all elements*,
        High Energy Density Physics I (2005),
        https://dx.doi.org/10.1016/j.hedp.2005.07.001

.. [SCFLY]
        H.-K. Chung, M.H. Chen, R.W. Lee.
        *Extension of atomic configuration sets of the Non-LTE model in the application to the Ka diagnostics of hot dense matter*,
        High Energy Density Physics III (2007),
        https://dx.doi.org/10.1016/j.hedp.2007.02.001
