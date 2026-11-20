#!/bin/bash
# BEGIN-PIC-DEPS-CHECK-LOCAL
# Verify the *_ROOT directories the bash preset's profile derives from PIC_LIBS.
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps check
# END-PIC-DEPS-CHECK-LOCAL
