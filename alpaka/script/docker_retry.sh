#!/bin/bash
#
# Copyright 2019-2020 Benjamin Worpitz, Rene Widera
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

ANSI_RED="\033[31m"
ANSI_RESET="\033[0m"

# rerun docker command if error 125 (
#   - triggered by image download problems
#   - wait 30 seconds before retry
docker_retry() {
  set +euo pipefail
  local result=0
  local count=1
  while [ $count -le 3 ]; do
    [ $result -eq 125 ] && {
      echo -e "\n${ANSI_RED}The command \"$*\" failed. Retrying, $count of 3.${ANSI_RESET}\n" >&2
    }
    "$@"
    result=$?
    [ $result -ne 125 ] && break
    count=$((count + 1))
    sleep 30
  done
  [ $count -gt 3 ] && {
    echo -e "\n${ANSI_RED}The command \"$*\" failed 3 times.${ANSI_RESET}\n" >&2
  }
  return $result
}
