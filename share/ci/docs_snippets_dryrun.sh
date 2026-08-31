#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2026 PIConGPU contributors
# Authors: opencode
# License: GPLv3+
#
# Local dry-run of the "docs-snippets" GitLab job.
#
# Extracts the job's `script:` entries from .gitlab-ci.yml exactly as the
# YAML parser (i.e. the GitLab runner) sees them, and runs the resulting
# script against the local checkout.
# Only the container-specific steps are stubbed:
#   - share/ci/install/pypicongpu.sh (micromamba environment; replaced by
#     the local python environment on PATH - the stub emulates its exports)
#   - apt (doxygen must already be installed locally)
#   - pic-build / pic-configure / tbg (only their --help smoke checks run)
# Everything else - in particular the snippet pytest suite, the profile
# check (share/ci/docs_snippets_profile_check.sh) and the doxygen +
# sphinx-build steps - runs for real.
#
# Requirements:
#   - a python environment with picongpu (pip install -e lib/python) and
#     docs/requirements.txt importable as `python3` on PATH
#   - doxygen on PATH
#
# Usage: share/ci/docs_snippets_dryrun.sh [path-to-repo-root]

set -euo pipefail

repo="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

# 1. extract the job script verbatim (the YAML-parsed script entries)
python3 - "$repo" "$work_dir" <<'PY'
import sys

import yaml

repo, work_dir = sys.argv[1], sys.argv[2]
with open(f"{repo}/.gitlab-ci.yml") as f:
    entries = yaml.safe_load(f)["docs-snippets"]["script"]
with open(f"{work_dir}/job.sh", "w") as f:
    for entry in entries:
        f.write(entry.rstrip("\n") + "\n")
PY

# 2. stubs for the container-specific steps
mkdir -p "$work_dir/stubs"
for tool in pic-build pic-configure tbg; do
    printf '#!/bin/sh\necho "dry-run stub: %s $*"\n' "$tool" > "$work_dir/stubs/$tool"
done
printf '#!/bin/sh\necho "dry-run stub: apt $*" >&2\n' > "$work_dir/stubs/apt"
chmod +x "$work_dir/stubs/"*

# 3. stub for share/ci/install/pypicongpu.sh (emulates its exports;
#    the local python environment replaces the micromamba one)
cat > "$work_dir/pypicongpu.sh" <<'EOF'
echo "dry-run stub: share/ci/install/pypicongpu.sh (local python environment)"
export PICSRC=$CI_PROJECT_DIR
export PATH=$PATH:$PICSRC/bin
EOF

# 4. the single deviation from the committed job script: source the stub
#    instead of the real environment-setup script
sed "s|^source \$CI_PROJECT_DIR/share/ci/install/pypicongpu.sh$|source $work_dir/pypicongpu.sh|" \
    "$work_dir/job.sh" > "$work_dir/job.dryrun.sh"
if grep -q '^source \$CI_PROJECT_DIR/share/ci/install/pypicongpu.sh$' "$work_dir/job.dryrun.sh"; then
    echo "ERROR: the pypicongpu.sh source line of the docs-snippets job changed;" >&2
    echo "       update the stub substitution in this dry-run script." >&2
    exit 1
fi

# 5. run the job script
cd "$work_dir"
export CI_PROJECT_DIR="$repo"
export PYTHON_VERSION="3.11.*"
export CI_CONTAINER_NAME="ubuntu24.04"
export PYPICONGPU_SKIP_QUICK_TEST=1
export PATH="$work_dir/stubs:$PATH"
bash job.dryrun.sh
