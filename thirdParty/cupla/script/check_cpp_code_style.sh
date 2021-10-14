#!/bin/bash

set -e
set -o pipefail

cd $CI_PROJECT_DIR

# check code style with clang format
find src example include test  -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.hpp" \
  | xargs clang-format-11 --dry-run --Werror
