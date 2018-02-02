#.rst:
# Findalpaka
# ----------
#
# Abstraction library for parallel kernel acceleration
# https://github.com/ComputationalRadiationPhysics/alpaka
#
# Finding and Using alpaka
# ^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: cmake
#
#   FIND_PACKAGE(alpaka
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.0.0
#     [REQUIRED]            # Fail with an error if alpaka or a required
#                           # component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: ignored
#   )
#   TARGET_LINK_LIBRARIES(<target> PUBLIC alpaka)
#
# To provide a hint to this module where to find the alpaka installation,
# set the ALPAKA_ROOT variable.
#
# This module requires Boost. Make sure to provide a valid install of it
# under the environment variable BOOST_ROOT.
#
# ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE will require Boost.Fiber to be built.
# ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE and ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE will require a OpenMP 2.0+ capable compiler.
# ALPAKA_ACC_CPU_BT_OMP4_ENABLE will require a OpenMP 4.0+ capable compiler.
# ALPAKA_ACC_GPU_CUDA_ENABLE will require CUDA 7.0+ to be installed.
# ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE will require TBB 2.2+ to be installed
#
# Set the following CMake variables BEFORE calling find_packages to
# change the behaviour of this module:
# - ``ALPAKA_ACC_GPU_CUDA_ONLY_MODE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_CPU_BT_OMP4_ENABLE`` {ON, OFF}
# - ``ALPAKA_ACC_GPU_CUDA_ENABLE`` {ON, OFF}
# - ``ALPAKA_CUDA_VERSION`` {7.0, ...}
# - ``ALPAKA_CUDA_ARCH`` {sm_20, sm...}
# - ``ALPAKA_CUDA_FAST_MATH`` {ON, OFF}
# - ``ALPAKA_CUDA_FTZ`` {ON, OFF}
# - ``ALPAKA_CUDA_SHOW_REGISTER`` {ON, OFF}
# - ``ALPAKA_CUDA_KEEP_FILES`` {ON, OFF}
# - ``ALPAKA_CUDA_SHOW_CODELINES`` {ON, OFF}
# - ``ALPAKA_DEBUG`` {0, 1, 2}
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# - ``alpaka_FOUND``
#   TRUE if alpaka found a working install.
# - ``alpaka_VERSION``
#   Version in format Major.Minor.Patch
# - ``alpaka_COMPILE_OPTIONS``
#   Compiler options.
# - ``alpaka_COMPILE_DEFINITIONS``
#   Compiler definitions (without "-D" prefix!).
# - ``alpaka_DEFINITIONS``
#   Deprecated old compiler definitions. Combination of alpaka_COMPILE_OPTIONS and alpaka_COMPILE_DEFINITIONS prefixed with "-D".
# - ``alpaka_INCLUDE_DIRS``
#   Include directories required by the alpaka headers.
# - ``alpaka_LIBRARIES``
#   Libraries required to link against to use alpaka.
#
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``alpaka``, if alpaka has
# been found.
#


################################################################################
# Copyright 2015 Benjamin Worpitz
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.

FIND_PATH(
    _ALPAKA_ROOT_DIR
    NAMES "include/alpaka/alpaka.hpp"
    HINTS "${ALPAKA_ROOT}" ENV ALPAKA_ROOT
    DOC "alpaka ROOT location")

IF(_ALPAKA_ROOT_DIR)
    INCLUDE("${_ALPAKA_ROOT_DIR}/alpakaConfig.cmake")
ELSE()
    MESSAGE(FATAL_ERROR "alpaka could not be found!")
ENDIF()
