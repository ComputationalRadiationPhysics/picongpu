#!/bin/bash
#
# Copyright 2023 Antonio Di Pilato, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

source ./script/travis_retry.sh

source ./script/set.sh

if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]
then
    # workaround to avoid link issues from python 2 to 3 during libomp dependency installation
    rm '/usr/local/bin/2to3-3.11' || true
    rm '/usr/local/bin/idle3.11' || true
    rm '/usr/local/bin/pydoc3.11' || true
    rm '/usr/local/bin/python3.11' || true
    rm '/usr/local/bin/python3.11-config' || true
    rm '/usr/local/bin/2to3-3.12'
    rm '/usr/local/bin/idle3.12'
    rm '/usr/local/bin/pydoc3.12'
    rm '/usr/local/bin/python3.12'
    rm '/usr/local/bin/python3.12-config'
    rm '/usr/local/bin/2to3' || true
    rm '/usr/local/bin/idle3' || true
    rm '/usr/local/bin/pydoc3' || true
    rm '/usr/local/bin/python3' || true
    rm '/usr/local/bin/python3-config' || true
    rm '/usr/local/share/man/man1/python3.1' || true
    rm '/usr/local/lib/pkgconfig/python3-embed.pc' || true
    rm '/usr/local/lib/pkgconfig/python3.pc' || true
    rm '/usr/local/Frameworks/Python.framework/Headers' || true
    rm '/usr/local/Frameworks/Python.framework/Python' || true
    rm '/usr/local/Frameworks/Python.framework/Resources' || true
    rm '/usr/local/Frameworks/Python.framework/Versions/Current'  || true
    brew reinstall --build-from-source --formula ./script/homebrew/${ALPAKA_CI_XCODE_VER}/libomp.rb
fi
