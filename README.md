PIConGPU - A particle-in-cell code for GPGPUs
================================================================

[![PIConGPU Presentation Video](http://img.youtube.com/vi/lgxVYl_pslI/0.jpg)](http://www.youtube.com/watch?v=lgxVYl_pslI)

Introduction
------------

PIConGPU is a relativistic
[Particle-in-Cell](http://en.wikipedia.org/wiki/Particle-in-cell)
(PIC) code running on
[graphic processing units](http://en.wikipedia.org/wiki/Graphics_processing_unit)
(GPUs). For Experts: PIConGPU is a fully 3D3V PIC-Code using various
numerical schemes. These include: a Yee-Lattice like grid
structure, particle pushers like the *Boris-* and the *Vay-Pusher*
and current deposition schemes like the ones proposed by
Villasenor-Buneman and Esirkepov. Macro-particle form factors
include NGP (0th order), CIC (1st), TSC(2nd) and PSQ (3rd).

PIConGPU is developed and maintained by the
[Junior Group Computational Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132&pOid=30354)
at the [Institute for Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132)
at [HZDR](http://www.hzdr.de/) in close collaboration with the Center
for Information Services and High Performance Computing
([ZIH](http://tu-dresden.de/die_tu_dresden/zentrale_einrichtungen/zih)) of the Technical University Dresden.

GPUs today reach a performance up to
[TFLOP/s](http://en.wikipedia.org/wiki/FLOPS)
at considerable lower invest and maintenance cost compared to CPU-based compute architectures of similar performance.

The Particle-in-Cell algorithm is a central tool in plasma physics.
It describes the dynamics of a plasma by computing the motion of
electrons and ions in the plasma based on
[Maxwell's equations](http://en.wikipedia.org/wiki/Maxwell%27s_equations).

Referencing & License
---------------------

PIConGPU is a *scientific project*. If you present and/or publish
scientific results that used PIConGPU, the according **publication** you should
set a **reference** to is:
- H Burau, et al,
  [PIConGPU : A Fully Relativistic Particle-in-Cell Code for a GPU Cluster](http://dx.doi.org/10.1109/TPS.2010.2064310),
  IEEE Transactions on Plasma Science 38(10), 2831-2839 (October 2010)

The following slide should be part of **oral presentations**. It is intended to
acknowledge the team maintaining PIConGPU and to support our community:
(*coming soon*) presentation_picongpu.pdf
(svg version, key note version, png version: 1920x1080 and 1024x768)

*PIConGPU* is licensed under the **GPLv3+**. You can use our *libraries* with **GPLv3+
or LGPLv3+** (they are *dual licensed*).
Please refer to our [LICENSE.md](LICENSE.md)

********************************************************************************

Install
-------

See our notes in [INSTALL.md](doc/INSTALL.md).

Users
-----

Visit [picongpu.hzdr.de](http://picongpu.hzdr.de) to learn more about PIC codes.
See the getting started guide (movie *coming soon*).

Developers
----------

### How to participate

See [PARTICIPATE.md](doc/PARTICIPATE.md)

Active Team
-----------

### Scientific Supervision

- Dr. Michael Bussmann
- Dr.-Ing. Guido Juckeland

### Maintainers* and core developers

- Heiko Burau*
- Anton Helm
- Axel Huebl*
- Richard Pausch*
- Felix Schmitt*
- Benjamin Schneider
- Rene Widera*

### Participants, Former Members and Thanks

The PIConGPU Team expresses its thanks to:

- Florian Berninger
- Robert Dietrich
- Wen Fu
- Wolfgang Hoehnig
- Remi Lehe
- Joseph Schuchart
- Klaus Steiniger

Kudos to everyone who helped!

********************************************************************************

![image of an lwfa](doc/images/lwfa_grey.png "LWFA")..
![image of our strong scaling](doc/images/StrongScalingPIConGPU_log.png "Strong Scaling")

![PIConGPU logo](doc/logo/pic_logo_320x140.png "PIConGPU")

