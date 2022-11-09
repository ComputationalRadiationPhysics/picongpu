Run time test for external beam
=================================

This test verifies the injection of particles (photons) at a boundary using the external beam module.
The particles are injection with a gaussian transversal profile and a square temporal pulse.
A python verifier compares the resulting particle per cell number (real particles not macro particles) with one computed in python.
This test covers the creation, positioning, and setting up initial momentum in the external beam module.

One way to run all tests in parallel:
.. code: bash
    for i in {0..6} ; do mkdir setup$i & done; wait;
    for i in {0..6} ; do python $PICSRC/share/picongpu/tests/ExternalBeam/lib/python/picongpu/tests/test_debug_external_beam.py -t $i --dir setup$i |& tee setup$i/test.log &  done; wait;
    for i in {0..6} ; do python $PICSRC/share/picongpu/tests/ExternalBeam/lib/python/picongpu/tests/verify_results.py  --dir setup$i/simOutput |& tee setup$i/verify.log &  done; wait;
