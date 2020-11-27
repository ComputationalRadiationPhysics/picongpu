mallocMC
=============

mallocMC: *Memory Allocator for Many Core Architectures*

This project provides a framework for **fast memory managers** on **many core
accelerators**. It is based on [alpaka](https://github.com/alpaka-group/alpaka)
to run on many different accelerators and implements the *ScatterAlloc* algorithm.


Usage
-------

Follow the step-by-step instructions in [Usage.md](Usage.md) to replace your
`new`/`malloc` calls with a *blacingly fast* mallocMC heap! :rocket:


Install
-------

mallocMC is header-only, but requires a few other C++ libraries to be
available. Our installation notes can be found in [INSTALL.md](INSTALL.md).


Contributing
------------

Rules for contributions are found in [CONTRIBUTING.md](CONTRIBUTING.md).

On the ScatterAlloc Algorithm
-----------------------------

This library implements the *ScatterAlloc* algorithm, originally
[forked](https://en.wikipedia.org/wiki/Fork_%28software_development%29)
from the **ScatterAlloc** project, developed by the
[Managed Volume Processing](http://www.icg.tugraz.at/project/mvp)
group at [Institute for Computer Graphics and Vision](http://www.icg.tugraz.at),
TU Graz (kudos!).

From http://www.icg.tugraz.at/project/mvp/downloads :
```quote
ScatterAlloc is a dynamic memory allocator for the GPU. It is
designed concerning the requirements of massively parallel
execution.

ScatterAlloc greatly reduces collisions and congestion by
scattering memory requests based on hashing. It can deal with
thousands of GPU-threads concurrently allocating memory and its
execution time is almost independent of the thread count.

ScatterAlloc is open source and easy to use in your CUDA projects.
```

Original Homepage: http://www.icg.tugraz.at/project/mvp

Our Homepage: https://www.hzdr.de/crp


Branches
--------

| *branch*    | *state* | *description*           |
| ----------- | ------- | ----------------------- |
| **master**  | [![Build Status Master](https://travis-ci.org/ComputationalRadiationPhysics/mallocMC.png?branch=master)](https://travis-ci.org/ComputationalRadiationPhysics/mallocMC "master") | our latest stable release |
| **dev**     | [![Build Status Development](https://travis-ci.org/ComputationalRadiationPhysics/mallocMC.png?branch=dev)](https://travis-ci.org/ComputationalRadiationPhysics/mallocMC "dev") | our development branch - start and merge new branches here |
| **tugraz**  | n/a | *ScatterAlloc* "upstream" branch: not backwards compatible mirror for algorithmic changes |



Literature
----------

Just an incomplete link collection for now:

- [Paper](http://www.icg.tugraz.at/Members/steinber/scatteralloc-1) by
  Markus Steinberger, Michael Kenzel, Bernhard Kainz and Dieter Schmalstieg

- 2012, May 5th: [Presentation](http://innovativeparallel.org/Presentations/inPar_kainz.pdf)
        at *Innovative Parallel Computing 2012* by *Bernhard Kainz*

- Junior Thesis [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.34461.svg)](http://dx.doi.org/10.5281/zenodo.34461) by
  Carlchristian Eckert (2014)


License
-------

We distribute the modified software under the same license as the
original software from TU Graz (by using the
[MIT License](https://en.wikipedia.org/wiki/MIT_License)).
Please refer to the [LICENSE](LICENSE) file.
