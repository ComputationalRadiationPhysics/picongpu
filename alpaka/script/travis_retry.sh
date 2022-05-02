#!/bin/bash
#
# MIT LICENSE
#
# Copyright (c) 2018 Travis CI GmbH <contact+travis-build@travis-ci.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

ANSI_RED="\033[31m"
ANSI_RESET="\033[0m"

travis_retry() {
  set +euo pipefail
  local result=0
  local count=1
  local max=666
  while [ $count -le $max ]; do
    [ $result -ne 0 ] && {
      echo -e "\n${ANSI_RED}The command \"$*\" failed. Retrying, $count of $max.${ANSI_RESET}\n" >&2
    }
    "$@"
    result=$?
    [ $result -eq 0 ] && break
    count=$((count + 1))
    sleep 1
  done
  [ $count -gt $max ] && {
    echo -e "\n${ANSI_RED}The command \"$*\" failed $max times.${ANSI_RESET}\n" >&2
  }
  return $result
}
