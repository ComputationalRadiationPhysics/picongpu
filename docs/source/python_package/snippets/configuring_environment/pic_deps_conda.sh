#!/bin/bash
# BEGIN-PIC-DEPS-CONDA
# Conda-backed local recipe: select the "bash/conda" preset in .picongpurc.toml
# (see rc_params_minimal_bash_conda.toml). pic-deps creates one conda-forge
# environment at $PIC_LIBS/conda, builds PNGwriter on top of it and reports the
# resolved versions.
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps install
# END-PIC-DEPS-CONDA
