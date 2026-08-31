#!/bin/bash
#
# Run the PIConGPU frontend quick test suite.
# This needs a source checkout of PIConGPU (the test tree is not part of
# the pip/uv installation); run it from the repository root.

cd lib/python/test/picongpu
python -m pytest quick/ -q
