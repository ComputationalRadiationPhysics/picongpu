# Copyright 2023 Rene Widera
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_GNU_INSTALL_ROOT} AND NOT DEFINED RISCV_GNU_INSTALL_ROOT)
    set(RISCV_GNU_INSTALL_ROOT "$ENV{RISCV_GNU_INSTALL_ROOT}" CACHE PATH "Path to GNU for RISC-V cross compiler installation directory")
else()
    set(RISCV_GNU_INSTALL_ROOT /opt/riscv CACHE PATH "Path to GNU for RISC-V cross compiler installation directory")
endif()
set(CMAKE_SYSROOT ${RISCV_GNU_INSTALL_ROOT}/sysroot CACHE PATH "RISC-V sysroot")

set(GNU_TARGET_TRIPLE riscv64-unknown-linux-gnu)

set(CMAKE_C_COMPILER ${RISCV_GNU_INSTALL_ROOT}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_C_COMPILER_TARGET ${GNU_TARGET_TRIPLE})
set(CMAKE_CXX_COMPILER ${RISCV_GNU_INSTALL_ROOT}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_CXX_COMPILER_TARGET ${GNU_TARGET_TRIPLE})
set(CMAKE_ASM_COMPILER ${RISCV_GNU_INSTALL_ROOT}/bin/riscv64-unknown-linux-gnu-as)
set(CMAKE_ASM_COMPILER_TARGET ${GNU_TARGET_TRIPLE})

# Avoids running the linker for source files passed to add_executable because cross-compiling required special
# linker flags.
# Attention, this can create issues during static MPI linking.
# A workaround is to pass linker option via `CMAKE_EXE_LINKER_FLAGS` and link this CMake variable
# explicit to target.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# prefer static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBRARIES OFF)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
