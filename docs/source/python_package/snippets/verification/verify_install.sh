#!/usr/bin/env bash
#
# Smoke-check the installed PIConGPU Python package:
# import the PICMI frontend and report its codename.
# Run this on any machine where you intend to use the package.

set -euo pipefail

python -c "from picongpu import picmi; print('PICMI codename:', picmi.codename)"
