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

# The submit step runs inside cwltool's per-step job cache directory
# (<run_dir>/.cwl_cache/<md5>) and bakes that path into tbg/submit.start
# (TBG_dstPath, --chdir) and link_results.sh.
# Rewrite the <...>/.cwl_cache/<md5> component back to the run directory it
# is derived from, so final outputs reference the stable run directory
# instead of the internal, potentially ephemeral cache.
for file in tbg/submit.start link_results.sh; do
    if [ -f "$file" ]; then
        sed -E -i 's|/\.cwl_cache/[0-9a-fA-F]+||g' "$file"
    fi
done
