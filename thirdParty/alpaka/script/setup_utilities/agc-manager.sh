#!/bin/bash

: ${ALPAKA_CI_OS_NAME?"ALPAKA_CI_OS_NAME must be specified"}

# the agc-manager only exists in the agc-container
# set alias to false, so each time if we ask the agc-manager if a software is installed, it will
# return false and the installation of software will be triggered
if [ "$ALPAKA_CI_OS_NAME" != "Linux" ] || [ ! -f "/usr/bin/agc-manager" ]; then
    # display message only one time not everytime the script is sourced
    if [ -z ${PRINT_INSTALL_AGC_MANAGER+x} ]; then
        echo_yellow "install fake agc-manager"
        export PRINT_INSTALL_AGC_MANAGER=true
    fi

    echo '#!/bin/bash' >agc-manager
    echo 'exit 1' >>agc-manager

    if [ "$ALPAKA_CI_OS_NAME" = "Linux" ]; then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/bin/agc-manager
    elif [ "$ALPAKA_CI_OS_NAME" = "Windows" ]; then
        chmod +x agc-manager
        mv agc-manager /usr/bin
    elif [ "$ALPAKA_CI_OS_NAME" = "macOS" ]; then
        sudo chmod +x agc-manager
        sudo mv agc-manager /usr/local/bin
    else
        echo_red "installing agc-manager: " \
        "unknown operation system: ${ALPAKA_CI_OS_NAME}"
        exit 1
    fi
else
    # display message only one time not everytime the script is sourced
    if [ -z ${PRINT_INSTALL_AGC_MANAGER+x} ]; then
        echo_green "found agc-manager"
        export PRINT_INSTALL_AGC_MANAGER=true
    fi
fi
