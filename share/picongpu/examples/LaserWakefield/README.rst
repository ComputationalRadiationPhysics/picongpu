LaserWakefield: Laser Electron Acceleration
===========================================

.. sectionauthor:: Axel Huebl <a.huebl (at) hzdr.de>
.. moduleauthor:: Axel Huebl <a.huebl (at) hzdr.de>, René Widera, Heiko Burau, Richard Pausch, Marco Garten

Setup for a laser-driven electron accelerator [TajimaDawson]_ in the blowout regime of an underdense plasma [Modena]_ [PukhovMeyerterVehn]_.
A short (fs) laser beam with ultra-high intensity (a_0 >> 1), modeled as a finite Gaussian beam is focussed in a hydrogen gas target.
The target is assumed to be pre-ionized with negligible temperature.
The relevant area of interaction is followed by a co-moving window, in whose time span the movement of ions is considered irrelevant which allows us to exclude those from our setup.

This is a demonstration setup to get a visible result quickly and test available methods and I/O.
The plasma gradients are unphysically high, the resolution of the laser wavelength is seriously bad, the laser parameters (e.g. pulse length, focusing) are challening to achieve technically and interaction region is too close to the boundaries of the simulation box.
Nevertheless, this setup will run on a single GPU in full 3D in a few minutes, so just enjoy running it and interact with our plugins!


References
----------

.. [TajimaDawson]
        T. Tajima, J.M. Dawson.
        *Laser electron accelerator*,
        Physical Review Letters (1979),
        https://dx.doi.org/10.1103/PhysRevLett.43.267

.. [Modena]
        A. Modena, Z. Najmudin, A.E. Dangor, C.E. Clayton, K.A. Marsh, C. Joshi, V. Malka, C. B. Darrow, C. Danson, D. Neely, F.N. Walsh.
        *Electron acceleration from the breaking of relativistic plasma waves*,
        Nature (1995),
        https://dx.doi.org/10.1038/377606a0

.. [PukhovMeyerterVehn]
        A. Pukhov and J. Meyer-ter-Vehn.
        *Laser wake field acceleration: the highly non-linear broken-wave regime*,
        Applied Physics B (2002),
        https://dx.doi.org/10.1007/s003400200795
