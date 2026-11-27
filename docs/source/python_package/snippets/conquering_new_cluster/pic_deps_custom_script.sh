#!/bin/bash
# BEGIN-PIC-DEPS-CUSTOM-SCRIPT
# Override the preset's installer with a local script for a single invocation.
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps check --script "$HOME/my_dependencies_autoinstall.sh"
uvx --from "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python" pic-deps install --script "$HOME/my_dependencies_autoinstall.sh"
# END-PIC-DEPS-CUSTOM-SCRIPT
