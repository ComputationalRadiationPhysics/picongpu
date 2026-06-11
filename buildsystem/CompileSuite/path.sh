#!/usr/bin/env bash

# SPDX-FileCopyrightText: Axel Huebl
#
# SPDX-License-Identifier: GPL-3.0-or-later

#
#

function absolute_path()
{
    cd $1
    pwd
}

function dir_exists()
{
    if [ ! -d "$1" ]; then
        echo "No $2 directory given." >&2
        exit 1
    fi
}
