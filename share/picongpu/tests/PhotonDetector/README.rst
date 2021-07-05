PhotonDetector:
===============
This is a set of compile tests as well as a series of runtime tests for the PhotonDetector plugin.
The simulation has two species.
Photons with a fixed wavelength, set with the wavelength flag, and photons with the `phase` attribute and the `PhotonPhase` pusher.
Each species has one particle.
The particles are initiated at one of the simulations boundaries and are send towards the detector.

The 3x3 detector can be placed behind any simulation boundary.

The compile tests are defined in cmakeFlags.
The runtime tests are defined in `lib/python/picongpu/tests` and can be run with `pytest`.
The runtime tests cover all possible detector placements as well as all detector cells for one placement, and different `openPMD` backends.

To run the tests:

#. Install all python requirements from `lib/python/picongpu/tests/requirements.txt`. For example with `conda create --name detector_tests --file <path to requirements.txt>`.
#. Source the picongpu profile (the PICSRC environment variable has to be set).
#. Run the tests with `pytest`.

When using a conda environment, don't activate the environment but rather explicitly specify the path to the pytest executable.
Otherwise, compiler may try to use the `openPMD-api` library from the python environment and not the one loaded in the `picongpu.profile`.
You may also need to empty `PYTHONPATH` or otherwise pytest may try to import the `openPMD-api` library that was meant for `picongpu`.

There is also an sbatch script in `etc/hemera-hzdr/run-tests_gpu_sbatch.sh` meant for the gpu partition on hemera.

.. sectionauthor:: Pawel Ordyna <p.ordyna (at) hzdr.de>
