#!/bin/bash
# create and enter a virtual environment first, e.g.
#   python -m venv .venv
#   source .venv/bin/activate
pip install "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
