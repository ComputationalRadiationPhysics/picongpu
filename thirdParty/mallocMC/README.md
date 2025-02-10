mallocMC
=============

mallocMC: *Memory Allocator for Many Core Architectures*

This project provides a framework for **fast memory managers** on **many core
accelerators**. It is based on [alpaka](https://github.com/alpaka-group/alpaka)
to run on many different accelerators and comes with multiple allocation
algorithms out-of-the-box. Custom ones can be added easily due to the
policy-based design.

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

Rules for contributions are found in [CONTRIBUTING.md](./CONTRIBUTING.md).

On the Algorithms
-----------------------------

This library was originally inspired by the *ScatterAlloc* algorithm,
[forked](https://en.wikipedia.org/wiki/Fork_%28software_development%29)
from the **ScatterAlloc** project, developed by the
[Managed Volume Processing](http://www.icg.tugraz.at/project/mvp)
group at [Institute for Computer Graphics and Vision](http://www.icg.tugraz.at),
TU Graz (kudos!). The currently shipped algorithms are using similar ideas but
differ from the original one significantly.

From the original project page (which is no longer existent to the best of our
knowledge):

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

Our Homepage: <https://www.hzdr.de/crp>

Versions and Releases
---------------------

Official releases can be found in the
[Github releases](https://github.com/alpaka-group/mallocMC/releases).
We try to stick to [semantic versioning](https://semver.org/) but we'll bump
the major version number for major features.
Development happens on the `dev` branch.
Changes there have passed the CI and a code review but we make no guarantees
about API or feature stability in this branch.

Literature
----------

Just an incomplete link collection for now:

- [Paper](https://doi.org/10.1109/InPar.2012.6339604) by
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
