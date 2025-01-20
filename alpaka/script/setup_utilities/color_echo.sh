#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# colored output

# display a message in green
echo_green() {
    # macOS uses bash 3, therefor \e is not working \033 needs to be used
    # https://stackoverflow.com/questions/28782394/how-to-get-osx-shell-script-to-show-colors-in-echo
    echo -e "\033[1;32m$1\033[0m"
}

# display a message in yellow
echo_yellow() {
    echo -e "\033[1;33m$1\033[0m"
}

# display a message in red
echo_red() {
    echo -e "\033[1;31m$1\033[0m"
}
