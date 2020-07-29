#!/bin/bash

set -e
set -o pipefail

# generate a reduced matrix with ci jobs based on the list (space separated) provided by the environment variable PIC_INPUTS

export PATH=$CI_PROJECT_DIR/share/ci:$PATH
export picongpu_DIR=$CI_PROJECT_DIR

cd $picongpu_DIR/share/picongpu/

echo "include:"
echo "  - local: '/share/ci/compiler_clang.yml'"
echo "  - local: '/share/ci/compiler_gcc.yml'"
echo "  - local: '/share/ci/compiler_nvcc_cuda.yml'"
echo "  - local: '/share/ci/compiler_clang_cuda.yml'"
echo ""

folders=()
for CASE in ${PIC_INPUTS}; do
  if [ "$CASE" == "examples" ] || [  "$CASE" == "tests"  ] ; then
      all_cases=$(find ${CASE}/* -maxdepth 0 -type d)
  else
      all_cases=$(find $CASE -maxdepth 0 -type d)
  fi
  for test_case_folder in $all_cases ; do
      folders+=($test_case_folder)
  done
done

echo "${folders[@]}" | tr " " "\n" | n_wise_generator.py $@
