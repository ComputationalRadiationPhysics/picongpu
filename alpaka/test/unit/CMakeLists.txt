#
# Copyright 2015-2022 Benjamin Worpitz, Jan Stephan, Antonio Di Pilato
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.18)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("acc/")
add_subdirectory("atomic/")
add_subdirectory("block/shared/")
add_subdirectory("block/sharedSharing/")
add_subdirectory("block/sync/")
add_subdirectory("core/")
add_subdirectory("dev/")
add_subdirectory("event/")
add_subdirectory("idx/")
add_subdirectory("intrinsic/")
add_subdirectory("kernel/")
add_subdirectory("math/")
add_subdirectory("mem/buf/")
add_subdirectory("mem/copy/")
add_subdirectory("mem/fence/")
add_subdirectory("mem/view/")
add_subdirectory("mem/p2p/")
add_subdirectory("meta/")
add_subdirectory("queue/")
add_subdirectory("rand/")
add_subdirectory("traits/")
add_subdirectory("vec/")
add_subdirectory("warp/")
add_subdirectory("workDiv/")
