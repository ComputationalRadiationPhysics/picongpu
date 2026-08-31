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
# (<run_dir>/.cwl_cache/<md5>) and can leave that path in tbg/submit.start
# and link_results.sh if a reference was not rewritten to the stable
# destination. As a backstop, strip the /.cwl_cache/<md5> path component,
# restoring the run directory it is derived from. cwltool cache keys are
# md5 (exactly 32 hex chars) and the component must end the path or be
# followed by '/', so a legitimate path that merely contains the string --
# e.g. a real directory named .cwl_cache with a non-32-hex entry, or a
# longer component sharing a hex prefix -- is left untouched. The trailing
# '/' (when present) is captured and re-emitted so the surrounding path
# components are not joined.
for file in tbg/submit.start link_results.sh; do
    if [ -f "$file" ]; then
        sed -E -i 's#/\.cwl_cache/[0-9a-fA-F]{32}(/|$)#\1#g' "$file"
    fi
done
