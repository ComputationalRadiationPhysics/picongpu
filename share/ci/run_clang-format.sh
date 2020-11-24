#!/bin/bash

set -e
set -o pipefail

cd $CI_PROJECT_DIR

find include/ share/picongpu/ share/pmacc -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
  -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
  -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
  | xargs clang-format-11 --dry-run --Werror
