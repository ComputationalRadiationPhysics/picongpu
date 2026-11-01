#!/bin/bash
# BEGIN-PIC-DEPS-SPACK
# Spack-backed local recipe: select the "bash/spack" preset in .picongpurc.toml
# (see rc_params_minimal_bash_spack.toml), then let pic-deps run the preset's
# Spack installer. Versions and *_ROOT paths come from the profile.
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps install
# END-PIC-DEPS-SPACK
