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
echo "  - local: '/share/ci/compiler_hipcc.yml'"
echo ""

# handle CI actions
has_label=$($CI_PROJECT_DIR/share/ci/pr_has_label.sh "CI:no-compile" && echo "0" || echo "1")
if [ "$has_label" == "0" ] ; then
  echo "skip-compile:"
  echo "  script:"
  echo "    - echo \"CI action - 'CI:no-compile' -> skip compile/runtime tests\""
  exit 0
else
  echo "Label 'CI:no-compile' for the current CI job not set." >&2
fi

if [ -n "$QUICK_CI_TESTS" ] ; then
  # For user PRs only run reduced set of tests.
  # If a PR is merged to the `dev` branch a non-reduced test will be executed.
  is_pr=$(echo "$CI_COMMIT_REF_NAME" | grep -q "^pr-" && echo 1 || echo 0)
  if [ $is_pr -eq 1 ] ; then
      ADDITIONAL_GENRATOR_FLAGS="--quick"
  fi
fi

folders=()
if [ "$PIC_INPUTS" == "pmacc" ] ; then
  # create test cases for PMacc
  echo "pmacc" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
elif [ "$PIC_INPUTS" == "pmacc_header" ] ; then
    # create test cases for PMacc
    echo "pmacc_header" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
elif [ "$PIC_INPUTS" == "unit" ] ; then
   # create test cases for PMacc
   echo "unit" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
else
  # create test cases for PIConGPU
  for CASE in ${PIC_INPUTS}; do
    if [ "$CASE" == "examples" ] || [  "$CASE" == "tests"  ] || [  "$CASE" == "benchmarks"  ] ; then
        all_cases=$(find ${CASE}/* -maxdepth 0 -type d)
    else
        all_cases=$(find $CASE -maxdepth 0 -type d)
    fi
    for test_case_folder in $all_cases ; do
        folders+=($test_case_folder)
    done
  done

  echo "${folders[@]}" | tr " " "\n" | n_wise_generator.py $@ $ADDITIONAL_GENRATOR_FLAGS
fi
