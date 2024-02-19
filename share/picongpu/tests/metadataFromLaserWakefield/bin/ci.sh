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

# doc-include-start: cmdline
pic-build && bin/picongpu -d 1 1 1 -g 24 24 24 --no-start-simulation --dump-metadata picongpu-metadata.json
# doc-include-end: cmdline

diff picongpu-metadata.json picongpu-metadata.json.reference
exit $?
