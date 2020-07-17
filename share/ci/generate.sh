#!/bin/bash

# generate a job matrix based on the environment variable lists (space separated)
# variables: GITLAB_BASES CXX_VERSIONS BOOST_VERSIONS PIC_INPUTS PIC_ACCS

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
