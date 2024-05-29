===========================
Test for openPMD-viewer API
===========================

This test reads out openPMD data via the openPMD-viewer and checks if it matches the input values to test if users can rely on the output.

To run this test, one has to execute ci.sh with the location of the input and output directory.

..code-block:: bash
./picongpu/share/picongpu/tests/openPMD-viewer-test/bin/ci.sh picongpu/share/picongpu/tests/openPMD-viewer-test/ ./run01
