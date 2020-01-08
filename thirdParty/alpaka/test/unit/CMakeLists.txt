#
# Copyright 2015-2019 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

################################################################################
# Add subdirectories.
################################################################################

ADD_SUBDIRECTORY("acc/")
ADD_SUBDIRECTORY("atomic/")
ADD_SUBDIRECTORY("block/shared/")
ADD_SUBDIRECTORY("block/sync/")
ADD_SUBDIRECTORY("core/")
ADD_SUBDIRECTORY("event/")
ADD_SUBDIRECTORY("idx/")
ADD_SUBDIRECTORY("kernel/")
ADD_SUBDIRECTORY("math/sincos/")
ADD_SUBDIRECTORY("mem/buf/")
ADD_SUBDIRECTORY("mem/view/")
ADD_SUBDIRECTORY("mem/p2p/")
ADD_SUBDIRECTORY("meta/")
ADD_SUBDIRECTORY("queue/")
ADD_SUBDIRECTORY("rand/")
ADD_SUBDIRECTORY("time/")
ADD_SUBDIRECTORY("vec/")
