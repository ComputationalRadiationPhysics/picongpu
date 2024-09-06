#!/bin/bash

set -e
set -o pipefail

# generate a reduced matrix with ci jobs based on the list (space separated) provided by the environment variable PIC_INPUTS

export PATH=$CI_PROJECT_DIR/share/ci:$PATH
export picongpu_DIR=$CI_PROJECT_DIR

cd $picongpu_DIR

# 0 == false; 1 == true
commit_no_compile=$(git log -1 | grep -q -i "^[[:blank:]]*ci:[[:blank:]]*no-compile[[:blank:]]*$" && echo "1" || echo "0")
commit_full_compile=$(git log -1 | grep -q -i "^[[:blank:]]*ci:[[:blank:]]*full-compile[[:blank:]]*$" && echo "1" || echo "0")
commit_picongpu_only=$(git log -1 | grep -q -i "^[[:blank:]]*ci:[[:blank:]]*picongpu[[:blank:]]*$" && echo "1" || echo "0")
commit_pmacc_only=$(git log -1 | grep -q -i "^[[:blank:]]*ci:[[:blank:]]*pmacc[[:blank:]]*$" && echo "1" || echo "0")

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
  echo "    - echo \"CI label action - 'CI:no-compile' -> skip compile/runtime tests\""
  exit 0
elif [ $commit_no_compile -eq 1 ]; then
  echo "skip-compile:"
  echo "  script:"
  echo "    - echo \"Commit CI control action - 'CI:no-compile' -> skip compile/runtime tests\""
  exit 0
else
  echo "Label 'CI:no-compile' for the current CI job not set." >&2
fi

if [ -n "$QUICK_CI_TESTS" ] ; then
  # For user PRs only run reduced set of tests.
  # If a PR is merged to the `dev` branch a non-reduced test will be executed.
  is_pr=$(echo "$CI_COMMIT_REF_NAME" | grep -q "^pr-" && echo 1 || echo 0)
  if [ $is_pr -eq 1 ] && [ $commit_full_compile -ne 1 ] ; then
      ADDITIONAL_GENRATOR_FLAGS="$ADDITIONAL_GENRATOR_FLAGS --quick"
  fi
fi

add_empty_job=0

folders=()
if [ "$PIC_INPUTS" == "pmacc" ] ; then
  if [ $commit_picongpu_only -ne 1 ] ; then
    # create unit tests for PMacc
    echo "pmacc" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
  else
    add_empty_job=1
  fi
elif [ "$PIC_INPUTS" == "pmacc_header" ] ; then
  if [ $commit_picongpu_only -ne 1 ] ; then
    # create header validation test for PMacc
    echo "pmacc_header" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
  fi
elif [ "$PIC_INPUTS" == "unit" ] ; then
  if [ $commit_pmacc_only -ne 1 ] ; then
    # create unit tests for PIConGPU
    echo "unit" | tr " " "\n" | n_wise_generator.py $@ --limit_boost_version $ADDITIONAL_GENRATOR_FLAGS
  else
    add_empty_job=1
  fi
else
  if [ $commit_pmacc_only -ne 1 ] ; then
    # create input set test cases for PIConGPU
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
  else
      add_empty_job=1
  fi
fi

if [ $add_empty_job -eq 1 ] ; then
  echo "skip-compile:"
  echo "  script:"
  echo "    - echo \"Commit CI control action - skip compile/runtime tests\""
  exit 0
fi
