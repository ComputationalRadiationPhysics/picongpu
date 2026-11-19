#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# If an error occurs (command does not return 0), try running the command again.
# Usage: retry_cmd command arg1 arg2 ...
#
# Configure variables
# - RETRY_CMD_MAX: number of retires (default 10)
# - RETRY_CMD_WAIT: wait N seconds between two tries (default 1)
# - RETRY_CONTINUE: If RETRY_CONTINUE=ON is set, the CI will continue instead stopping via
#       exit_error().
retry_cmd() {
    if [[ $# -lt 1 ]]; then
        exit_error "retry_cmd requires at least one argument."
    fi

    echo_green "$*"
    (
        set +euo pipefail
        local max_tries="${RETRY_CMD_MAX:-10}"

        # time in seconds
        local wait_time="${RETRY_CMD_WAIT:-1}"

        for ((i = 0; i < max_tries; ++i)); do
            "$@"
            result="$?"

            if [[ "$result" -eq 0 ]]; then
                return 0
            fi

            echo_yellow "[WARNING]: Attempt #${i} to run '$*' failed"
            sleep "$wait_time"
        done
        if [[ -z ${RETRY_CONTINUE+x} ]] || [[ "${RETRY_CONTINUE}" != "ON" ]]; then
            echo_red "run '$*' failed" "$result"
            exit "$result"
        else
            return "$result"
        fi
    )
}
