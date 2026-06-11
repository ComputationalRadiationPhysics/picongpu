# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_GNU_INSTALL_ROOT} AND NOT DEFINED RISCV_GNU_INSTALL_ROOT)
    set(RISCV_GNU_INSTALL_ROOT
        "$ENV{RISCV_GNU_INSTALL_ROOT}"
        CACHE PATH
        "Path to GNU for RISC-V cross compiler installation directory"
    )
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
