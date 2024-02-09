#!/bin/bash

set -e
set -o pipefail

cd $CI_PROJECT_DIR

#############################################################################
# Conformance with Alpaka: Do not write __global__ CUDA kernels directly    #
#############################################################################
test/hasCudaGlobalKeyword include/pmacc
test/hasCudaGlobalKeyword share/pmacc/examples
test/hasCudaGlobalKeyword include/picongpu
test/hasCudaGlobalKeyword share/picongpu/examples

#############################################################################
# Enforce angle brackets <...> for includes of external library files       #
#############################################################################
test/hasExtLibIncludeBrackets include boost
test/hasExtLibIncludeBrackets include alpaka
test/hasExtLibIncludeBrackets include cupla
test/hasExtLibIncludeBrackets include mallocMC
test/hasExtLibIncludeBrackets include/picongpu pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples boost
test/hasExtLibIncludeBrackets share/picongpu/examples alpaka
test/hasExtLibIncludeBrackets share/picongpu/examples cupla
test/hasExtLibIncludeBrackets share/picongpu/examples mallocMC
test/hasExtLibIncludeBrackets share/pmacc/examples pmacc

#############################################################################
# Disallow doxygen with \                                                   #
#############################################################################
test/hasWrongDoxygenStyle include param
test/hasWrongDoxygenStyle include tparam
test/hasWrongDoxygenStyle include see
test/hasWrongDoxygenStyle include return
test/hasWrongDoxygenStyle include treturn
test/hasWrongDoxygenStyle share param
test/hasWrongDoxygenStyle share tparam
test/hasWrongDoxygenStyle share see
test/hasWrongDoxygenStyle share return
test/hasWrongDoxygenStyle share treturn
