#!/bin/bash
# BEGIN-PIC-DEPS-INSTALL-LOCAL
# Build PIConGPU's dependencies into rc_params['pic_libs'] for the local
# (module-less) "bash" preset.
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps install
# END-PIC-DEPS-INSTALL-LOCAL
