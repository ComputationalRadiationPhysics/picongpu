#!/bin/bash

# inside the agc-container, the user is root and does not require sudo
# to compatibility to other container, fake the missing sudo command
if ! command -v sudo &>/dev/null; then
    if [ "$ALPAKA_CI_OS_NAME" == "Linux" ]; then
        # display message only one time not everytime the script is sourced
        if [ -z ${PRINT_INSTALL_SUDO+x} ]; then
            echo_yellow "install sudo"
            export PRINT_INSTALL_SUDO=true
        fi

        DEBIAN_FRONTEND=noninteractive travis_retry apt update
        DEBIAN_FRONTEND=noninteractive travis_retry apt install -y sudo
    fi
fi
