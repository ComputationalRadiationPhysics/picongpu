#!/usr/bin/env bash
#
# Manage a job that was submitted from a run directory:
# read the job id, cancel the job and link the results.
#
# Usage: manage_submission.sh <run_dir> [link_name]
#
# Cancel the job with the command of your batch system
# (scancel for Slurm, qdel for PBS/LSF, ...).

set -euo pipefail

run_dir="$1"
link_name="${2:-results}"

# the job id (or, for the "bash" submit system, the shell PID)
job_id="$(tr -cd '0-9' < "$run_dir/submission_information.txt" | head -c 10)"

scancel "$job_id"

# link the output of the job into the run directory
"$run_dir/link_results.sh" "$run_dir/$link_name"
