#!/bin/bash

set -e
set -o pipefail

cd $CI_PROJECT_DIR

# check code style with clang format
find include/ share/picongpu/ share/pmacc -iname "*.def" \
  -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
  -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
  -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
  | xargs clang-format-11 --dry-run --Werror

#############################################################################
# Conformance with Alpaka: Do not write __global__ CUDA kernels directly    #
#############################################################################
test/hasCudaGlobalKeyword include/pmacc
test/hasCudaGlobalKeyword share/pmacc/examples
test/hasCudaGlobalKeyword include/picongpu
test/hasCudaGlobalKeyword share/picongpu/examples

#############################################################################
# Disallow end-of-line (EOL) white spaces                                   #
#############################################################################
test/hasEOLwhiteSpace

#############################################################################
# Disallow TABs, use white spaces                                           #
#############################################################################
test/hasTabs

#############################################################################
# Disallow non-ASCII in source files and scripts                            #
#############################################################################
test/hasNonASCII

#############################################################################
# Disallow spaces before pre-compiler macros                                #
#############################################################################
test/hasSpaceBeforePrecompiler

#############################################################################
# Enforce angle brackets <...> for includes of external library files       #
#############################################################################
test/hasExtLibIncludeBrackets include boost
test/hasExtLibIncludeBrackets include alpaka
test/hasExtLibIncludeBrackets include cupla
test/hasExtLibIncludeBrackets include splash
test/hasExtLibIncludeBrackets include mallocMC
test/hasExtLibIncludeBrackets include/picongpu pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples boost
test/hasExtLibIncludeBrackets share/picongpu/examples alpaka
test/hasExtLibIncludeBrackets share/picongpu/examples cupla
test/hasExtLibIncludeBrackets share/picongpu/examples splash
test/hasExtLibIncludeBrackets share/picongpu/examples mallocMC
test/hasExtLibIncludeBrackets share/pmacc/examples pmacc
