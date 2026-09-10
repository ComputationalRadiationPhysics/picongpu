#!/bin/bash
set -euxo pipefail

PROJECT_PATH="$1"
BIN_DIRECTORY="$2"
TBG_DIRECTORY="$3"
SUBMISSION_INFORMATION="$4"
LINK_RESULTS_SCRIPT="$5"

cp -r "$PROJECT_PATH" input
cp -r "$BIN_DIRECTORY" input/bin

cp -r "$TBG_DIRECTORY" tbg
cp "$SUBMISSION_INFORMATION" submission_information.txt
cp "$LINK_RESULTS_SCRIPT" link_results.sh

# Every workflow step runs in isolation inside cwltool's per-step job cache
# directory (<run_dir>/.cwl_cache/<md5>), and the submit step bakes that
# directory into its outputs (in tbg/submit.start, TBG_dstPath, --chdir and
# the executable path all point into the cache). To let the final run_dir be
# treated as if the simulation had run there directly (legacy behaviour)
# while keeping CWL's step isolation intact, strip every reference to the
# internal cwltool job cache from all generated files -- except
# link_results.sh, which is the one file allowed to keep pointing at the
# cache, because that is where the isolated job actually wrote its results.
#
# cwltool cache keys are md5 (exactly 32 hex chars); the component must end
# the path or be followed by '/', so an unrelated path that merely contains
# the string (e.g. a real directory named .cwl_cache holding a non-32-hex
# entry) is left untouched. The trailing '/' (when present) is captured and
# re-emitted so neighbouring path components are not joined.
#
# `grep -RlZEI` lists (NUL-separated) only the regular *text* files that
# actually contain a reference, skipping binary files (the compiled
# executable under input/bin) and excluding link_results.sh and the cache
# directory itself; `xargs -0 -r` then strips the reference from just those
# files. `-R` (not just `-r`) is essential: cwltool stages intermediate step
# outputs (tbg, input) as symlinks in this step's working directory, and
# `grep -r` does not recurse through symlinks to directories, so the
# references they hold would be missed. `|| true` keeps `set -e`/`pipefail`
# from aborting when nothing matches (e.g. a run that already points at a
# stable directory).
grep -RlZEI --exclude='link_results.sh' --exclude-dir='.cwl_cache' '/\.cwl_cache/[0-9a-fA-F]{32}($|/)' . \
    | xargs -0 -r sed -E -i 's#/\.cwl_cache/[0-9a-fA-F]{32}(/|$)#\1#g' || true
