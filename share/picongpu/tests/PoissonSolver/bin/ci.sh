#!/usr/bin/env bash
#
# Run the Gauss-Law checks and the analytic regression cases.
#
# Usage:
#   ci.sh <input-set> <output-dir>


set -uo pipefail

help()
{
    echo "Run the gauss-law Poisson checks and the three analytic cases."
    echo
    echo "Usage: $0 <input-set> <output-dir>"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    help
    exit 0
fi

if [ $# -ne 2 ]; then
    help >&2
    exit 2
fi

current_input=$(cd -- "$1" && pwd)
output_dir=$(realpath -m -- "$2")
analytic_input="$current_input/analytic_V"
gauss_law_input="$current_input/gauss_law"

if [[ ! -d "$gauss_law_input/include" ]]; then
    echo "Gauss-Law input set has no include directory: $gauss_law_input" >&2
    exit 2
fi

if [[ ! -d "$analytic_input/include" ]]; then
    echo "Analytic input set has no include directory: $analytic_input" >&2
    exit 2
fi

if [[ -e "$output_dir" ]]; then
    echo "Output directory already exists; removing now: $output_dir" >&2
    rm -r "$output_dir"
fi
mkdir -p -- "$output_dir"

gauss_law_build="$output_dir/gauss-law"
analytic_neumann_build="$output_dir/analytic-neumann"
analytic_dirichlet_build="$output_dir/analytic-dirichlet"

run_logged()
{
    local log_file=$1
    shift

    "$@" 2>&1 | tee "$log_file"
    local -a pipeline_status=( "${PIPESTATUS[@]}" )

    if (( pipeline_status[0] != 0 )); then
        return "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        echo "Writing log failed: $log_file" >&2
        return "${pipeline_status[1]}"
    fi
}

run_logged_cd()
{
    local log_file=$1
    local dir=$2
    shift 2

    (
      cd -- "$dir" || exit 1
      run_logged "$log_file" "$@"
    )
}

run_checked()
{
    local description=$1
    shift

    echo "$description"
    "$@"
    local status=$?
    if (( status != 0 )); then
        echo "$description failed (exit $status)" >&2
        exit "$status"
    fi
}

echo "Creating input sets"
run_checked "Creating gauss-law input set" \
    pic-create "$gauss_law_input" "$gauss_law_build"
run_checked "Creating analytic Neumann input set" \
    pic-create "$analytic_input" "$analytic_neumann_build"
run_checked "Creating analytic Dirichlet input set" \
    pic-create "$analytic_input" "$analytic_dirichlet_build"

echo "Building input sets"
run_checked "Building gauss-law input set" \
    bash -c 'cd "$1" && pic-build' _ "$gauss_law_build"
run_checked "Building analytic Neumann input set" \
    bash -c 'cd "$1" && pic-build' _ "$analytic_neumann_build"

# The analytic poissonSolver.param must use this macro to select Dirichlet.
run_checked "Building analytic Dirichlet input set" \
    bash -c 'cd "$1" && pic-build -c "$2"' _ \
    "$analytic_dirichlet_build" \
    "-DPARAM_OVERWRITES=-DPARAM_POISSON_BOUNDARY_CONDITION_DIRICHLET"

# gauss-law tests: keep their original grid, step count, output names and validator.
gauss_law_sim="$output_dir/gauss-law-sim"
mkdir -p -- "$gauss_law_sim"

run_checked "Running gauss-law open-boundary case" \
    run_logged_cd "$gauss_law_sim/open.log" "$gauss_law_sim" \
    mpiexec -n 1 "$gauss_law_build/bin/picongpu" \
        -d 1 1 1 -g 128 512 128 --periodic 0 0 0 -s 1 \
        --openPMD.period 8 --openPMD.ext bp --openPMD.file simDataOpen

run_checked "Validating gauss-law open-boundary case" \
    python3 "$gauss_law_build/lib/python/test/validate_results.py" \
        -r "$gauss_law_sim" -b open

run_checked "Running gauss-law periodic case" \
    run_logged_cd "$gauss_law_sim/periodic.log" "$gauss_law_sim" \
    mpiexec -n 1 "$gauss_law_build/bin/picongpu" \
        -d 1 1 1 -g 128 512 128 --periodic 1 1 1 -s 1 \
        --openPMD.period 8 --openPMD.ext bp --openPMD.file simDataPeriodic

run_checked "Validating gauss-law periodic case" \
    python3 "$gauss_law_build/lib/python/test/validate_results.py" \
        -r "$gauss_law_sim" -b periodic

# Analytic cases share a results directory but use unique openPMD file names.
analytic_sim="$output_dir/analytic-sim"
mkdir -p -- "$analytic_sim"

run_analytic_case()
{
    local case_name=$1
    local binary=$2
    local periodic_x=$3
    local periodic_y=$4
    local periodic_z=$5
    local validator_case=$6
    local file_base="simDataCase${case_name}"
    local log_file="$analytic_sim/poissonCase${case_name}.log"

    run_checked "Running analytic case $case_name" \
        run_logged_cd "$log_file" "$analytic_sim" \
        mpiexec -n 1 "$binary" \
            -d 1 1 1 -g 32 32 32 \
            --periodic "$periodic_x" "$periodic_y" "$periodic_z" \
            -s 0 \
            --openPMD.period 1 --openPMD.ext bp \
            --openPMD.file "$file_base" -r 2

    run_checked "Checking boundary-condition log for case $case_name" \
        check_analytic_bc "$case_name" "$log_file"

    run_checked "Validating analytic case $case_name" \
        python3 "$analytic_neumann_build/lib/python/test/validate_analytic.py" \
            -r "$analytic_sim" -f "$file_base" -c "$validator_case"
}

check_analytic_bc()
{
    local case_name=$1
    local log_file=$2
    local expected_bc
    local expect_mean

    case "$case_name" in
        A)
            expected_bc="used boundary conditions: homogeneous Neumann"
            expect_mean=yes
            ;;
        B|C)
            expected_bc="used boundary conditions: homogeneous Dirichlet"
            expect_mean=no
            ;;
        *)
            echo "Unknown analytic case: $case_name" >&2
            return 2
            ;;
    esac

    if ! grep -qF -- "$expected_bc" "$log_file"; then
        echo "Case $case_name did not report '$expected_bc'." >&2
        return 1
    fi

    if [[ "$expect_mean" == yes ]]; then
        if ! grep -qF -- "offsetting mean of rho by:" "$log_file"; then
            echo "Case $case_name should report the Neumann mean projection." >&2
            return 1
        fi
    elif grep -qF -- "offsetting mean of rho by:" "$log_file"; then
        echo "Case $case_name unexpectedly projected the source mean." >&2
        return 1
    fi
}

# A: fully periodic sine mode, Neumann build.
run_analytic_case A "$analytic_neumann_build/bin/picongpu" 1 1 1 A

# B: grounded Dirichlet box; C: Dirichlet in x, periodic in y and z.
run_analytic_case B "$analytic_dirichlet_build/bin/picongpu" 0 0 0 B
run_analytic_case C "$analytic_dirichlet_build/bin/picongpu" 0 1 1 C

echo "All existing and analytic Poisson tests passed."
