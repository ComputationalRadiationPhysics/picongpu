#!/bin/bash

# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
