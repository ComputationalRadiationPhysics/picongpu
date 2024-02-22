#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2024 PIConGPU contributors
# Authors: Julian Lenz
# License: GPLv3+
#

set -e
set -o pipefail

function absolute_path() {
    builtin cd -- "$1" &>/dev/null && pwd
}

help() {
    echo "Output the metadata from the LaserWakefield example and compare it to a reference file."
    echo ""
    echo "Usage:"
    echo "    (1) Change current working directory to directory where the include directory of the setup is located"
    echo "    (2) execute ci.sh from this directory"
    echo ""
    echo "Options"
    echo "-h | --help                   - show help"
    echo ""
}

#####################
## option handling ##
#####################
# options may be followed by
# - one colon to indicate they have a required argument
OPTS=$(getopt -o h -l help -- "$@")
if [ $? != 0 ]; then
    # something went wrong, getopt will put out an error message for us
    exit 1
fi

eval set -- "$OPTS"

# parser
while true; do
    case "$1" in
    -h | --help)
        echo -e "$(help)"
        shift
        exit 0
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

############################
## build and run picongpu ##
############################
if ! [ -d "./include" ]; then
    echo "Execute ci.sh from the directory where the simulation include dir is located!"
    exit 1
fi

OPTIONAL_FILENAME="picongpu-metadata.json"
REFERENCE_FILE="picongpu-metadata.json.reference"
TMPDIR="$(mktemp -d)"

if ! [ -d "${TMPDIR}" ]; then
    echo "Creation of temporary directory failed."
    exit 1
fi

function cleanup {
    rm -rf "${TMPDIR}"
}

trap cleanup EXIT

cp "${REFERENCE_FILE}" "${TMPDIR}/${REFERENCE_FILE}"

pic-create -f . "${TMPDIR}" &&
    cd "${TMPDIR}" &&
    pic-build

EXECUTABLE="bin/picongpu"
ARGS="-d 1 1 1 -g 24 24 24 "

# doc-include-start: cmdline
${EXECUTABLE} ${ARGS} --dump-metadata "${OPTIONAL_FILENAME}" --no-start-simulation
# doc-include-end: cmdline

diff "${OPTIONAL_FILENAME}" "${REFERENCE_FILE}"
exit $?
