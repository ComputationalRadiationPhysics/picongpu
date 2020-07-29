#!/bin/bash

set -e
set -o pipefail

# generate a job matrix based on the environment variable lists (space separated)
# hidden base job to generate the test matrix
# required variables (space separated lists):
#   PIC_INPUTS - path to examples relative to share/picongpu
#                e.g.
#                    "examples" starts one gitlab job per directory in `examples/*`
#                    "examples/" compile all directories in `examples/*` within one gitlab job
#                    "examples/KelvinHelmholtz" compile all cases within one gitlab job
#   GITLAB_BASES   - name of the hidden gitlab base job desctption `/share/ci/compiler_*`
#   CXX_VERSIONS   - name of the compiler to use see `/share/ci/compiler_*` `e.g. "g++-8 g++-6"
#   BOOST_VERSIONS - boost version to check e.g. "1.70.0"
#                    supported version: {1.65.1, 1.66.0, 1.67.0, 1.68.0, 1.69.0, 1.70.0, 1.71.0, 1.72.0, 1.73.0}
#   PIC_ACCS       - PIConGPU backend names see `pic-build --help`
#                    e.g. "cuda cuda:35 serial"

export picongpu_DIR=$CI_PROJECT_DIR
cd $picongpu_DIR/share/picongpu/

echo "include:"
echo "  - local: '/share/ci/compiler_clang.yml'"
echo "  - local: '/share/ci/compiler_gcc.yml'"
echo "  - local: '/share/ci/compiler_nvcc_cuda.yml'"
echo "  - local: '/share/ci/compiler_clang_cuda.yml'"
echo ""

for base_job in $GITLAB_BASES; do
    for CXX_VERSION in $CXX_VERSIONS; do
        for BOOST_VERSION in ${BOOST_VERSIONS}; do
            for CASE in ${PIC_INPUTS}; do
                for ACC in ${PIC_ACCS}; do
                    if [ "$CASE" == "examples" ] || [  "$CASE" == "tests"  ] ; then
                        all_cases=$(find ${CASE}/* -maxdepth 0 -type d)
                    else
                        all_cases=$(find $CASE -maxdepth 0 -type d)
                    fi
                    for test_case_folder in $all_cases ; do
                        job_name="${CXX_VERSION}_${ACC}_${BOOST_VERSION}_$(echo $test_case_folder | tr '/' '.')"
                        echo "${job_name}:"
                        echo "  variables:"
                        echo "    PIC_TEST_CASE_FOLDER: \"${test_case_folder}\""
                        echo "    CXX_VERSION: \"${CXX_VERSION}\""
                        echo "    PIC_BACKEND: \"${ACC}\""
                        echo "    BOOST_VERSION: \"${BOOST_VERSION}\""
                        echo "  before_script:"
                        echo "    - apt-get update -qq"
                        echo "    - apt-get install -y -qq libopenmpi-dev openmpi-bin openssh-server"
                        echo "  extends: $base_job"
                        echo ""
                    done
                done
            done
        done
    done
done
