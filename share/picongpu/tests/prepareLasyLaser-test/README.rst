===========================
Test for prepareLasyLaser module
===========================

This test generates a Lasy laser, saves it to openPMD and imports it to PIConGPU as incidentField using FromOpenPMDPulse and tests if all that works.

To run this test, one has to execute ci.sh with the location of the input and output directory.

..code-block:: bash
./picongpu/share/picongpu/tests/prepareLasyLaser-test/bin/ci.sh picongpu/share/picongpu/tests/prepareLasyLaser-test/ ./run02
