#!/bin/bash

set -e
set -o pipefail

if [ "$1" == "pmacc" ] ; then
  $CI_PROJECT_DIR/share/ci/run_pmacc_tests.sh
elif [ "$1" == "unit" ] ; then
    $CI_PROJECT_DIR/share/ci/run_picongpu_unit_tests.sh
else
  $CI_PROJECT_DIR/share/ci/run_picongpu_tests.sh
fi
