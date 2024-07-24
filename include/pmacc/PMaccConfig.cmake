# Copyright 2015-2023 Erik Zenker, Rene Widera, Axel Huebl, Jan Stephan
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#


# - Config file for the pmacc package and provide the target pmacc::pmacc

###############################################################################
# PMacc
###############################################################################
cmake_minimum_required(VERSION 3.25.0)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

# own modules for find_packages e.g. FindmallocMC
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PMacc_DIR}/../../thirdParty/cmake-modules/)

################################################################################
# alpaka path
################################################################################

# workaround for native CMake CUDA
# CMake is not forwarding CMAKE_CUDA_ARCHITECTURES to the CMake CUDA compiler check
# error: clang: error: cannot find libdevice for sm_20. Provide path to different CUDA installation via --cuda-path, or pass -nocudalib to build without linking with libdevice.
# The workaround is parsing CMAKE_CUDA_ARCHITECTURES and forward command line parameter directly to clang++.
if(alpaka_ACC_GPU_CUDA_ENABLE AND CMAKE_CUDA_COMPILER)
    string(REGEX MATCH "(.*clang.*)" IS_CLANGCUDA_COMPILER ${CMAKE_CUDA_COMPILER})
    if(IS_CLANGCUDA_COMPILER)
        foreach(_CUDA_ARCH_ELEM ${CMAKE_CUDA_ARCHITECTURES})
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=sm_${_CUDA_ARCH_ELEM}")
        endforeach()
    endif()
endif()

# workaround for a CMake bug which is not handled in alpaka 0.7.0
# https://github.com/alpaka-group/alpaka/pull/1423
if(alpaka_ACC_GPU_CUDA_ENABLE)
    include(CheckLanguage)
    check_language(CUDA)
    # Use user selected CMake CXX compiler as cuda host compiler to avoid fallback to the default system CXX host compiler.
    # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definition passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
    # ignored by CMake (looks like a CMake bug).
    # The if condition used here should work correct after the CMake bug is fixed, too.
    # Check the environment variable CUDAHOSTCXX to prefer the CUDA host compiler set by the user.
    if("$ENV{CUDAHOSTCXX}" STREQUAL "" AND NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
    enable_language(CUDA)
endif()

# set path to internal
set(PMACC_alpaka_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE PMACC_alpaka_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(PMACC_alpaka_PROVIDER)

if(${PMACC_alpaka_PROVIDER} STREQUAL "intern")
    list(INSERT CMAKE_MODULE_PATH 0 "${PMacc_DIR}/../../thirdParty/alpaka")
endif()

# Set alpaka CXX standard because the default is currently C++14.
if(NOT DEFINED alpaka_CXX_STANDARD)
    set(alpaka_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "C++ standard version")
endif()

################################################################################
# setup alpaka
################################################################################

# the min and max. supported alpaka version
set(_PMACC_MIN_ALPAKA_VERSION 1.2.0)
set(_PMACC_MAX_ALPAKA_VERSION 1.2.0)

# do not search for alpaka if it already exists
# for example, a project that includes alpaka via add_subdirectory before including pmacc via add_subdirectory
if(NOT TARGET alpaka::alpaka)
    # the alpaka provider for the internal alpaka is only available,
    # if pmacc is used via add_subdirectory in another project
    # or examples are build

    set(PMACC_ALPAKA_PROVIDER "internal" CACHE STRING "Select which alpaka is used")
    set_property(CACHE PMACC_ALPAKA_PROVIDER PROPERTY STRINGS "internal;external")
    mark_as_advanced(PMACC_ALPAKA_PROVIDER)

    if(${PMACC_ALPAKA_PROVIDER} STREQUAL "internal")
        set(alpaka_BUILD_EXAMPLES OFF)
        set(BUILD_TESTING OFF)
        add_subdirectory(${PMacc_DIR}/../../thirdParty/alpaka ${CMAKE_BINARY_DIR}/alpaka)
    else()
        find_package(alpaka ${_PMACC_MAX_ALPAKA_VERSION} HINTS $ENV{ALPAKA_ROOT})
        if(NOT TARGET alpaka::alpaka)
            message(STATUS "Could not find alpaka ${_PMACC_MAX_ALPAKA_VERSION}. Now searching for alpaka ${_PMACC_MIN_ALPAKA_VERSION}")
            find_package(alpaka ${_PMACC_MIN_ALPAKA_VERSION} REQUIRED HINTS $ENV{ALPAKA_ROOT})
        endif()
        if(alpaka_VERSION VERSION_GREATER _PMACC_MAX_ALPAKA_VERSION)
            message(WARNING "Unsupported alpaka version ${alpaka_VERSION}. "
                    "Supported versions [${_PMACC_MIN_ALPAKA_VERSION},${_PMACC_MAX_ALPAKA_VERSION}].")
        endif()
    endif()

    if(NOT TARGET alpaka::alpaka)
        message(FATAL_ERROR "Required PMacc dependency alpaka could not be found!")
    endif()
endif()

################################################################################
# PMacc target
################################################################################

file(GLOB_RECURSE PMACC_SRC_FILES "${PMacc_DIR}/*.cpp")
# remove files located in the directory 'test'
string(REGEX REPLACE "${PMacc_DIR}/test/.*" "" PMACC_SRC_FILES "${PMACC_SRC_FILES}")

alpaka_add_library(
        pmacc
        STATIC
        ${PMACC_SRC_FILES}
)

target_include_directories(pmacc
        PUBLIC
        $<BUILD_INTERFACE:${PMacc_DIR}/..>
        $<INSTALL_INTERFACE:${PMacc_DIR}/..>)

# Even if there are no sources CMAKE has to know the language.
set_target_properties(pmacc PROPERTIES LINKER_LANGUAGE CXX)

add_library(pmacc::pmacc ALIAS pmacc)
target_link_libraries(pmacc PUBLIC alpaka::alpaka)

# Create traget pmacc::filesystem to handle system where std::filesystem is experimental.
# e.g. on systems with an old libstdc++ <= 7
add_library(pmacc_filesystem INTERFACE)
add_library(pmacc::filesystem ALIAS pmacc_filesystem)
target_link_libraries(pmacc PUBLIC pmacc::filesystem)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(std::filesystem::path::preferred_separator "filesystem" PMACC_FOUND_CXX17_STD_FILESYSTEM)
if(NOT PMACC_FOUND_CXX17_STD_FILESYSTEM)
    message(STATUS "Switch using 'std::experimental::filesystem'")
    target_compile_definitions(pmacc_filesystem INTERFACE "-DPMACC_USE_STD_EXPERIMENTAL_FILESYSTEM=1")
    target_link_libraries(pmacc_filesystem INTERFACE stdc++fs)
endif()


###############################################################################
# Build Flags
###############################################################################

set(PMACC_BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type for the project" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${PMACC_BUILD_TYPE}")
unset(PMACC_BUILD_TYPE)


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


###############################################################################
# Definitely Unsupported Compilers
###############################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "GCC too old! Use GCC 7.0 or newer")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(WARNING "GCC < 9.0 is not a official supported, please try a newer!")
    endif()
# Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "Apple Clang (Xcode) too old! Use Xcode 8.0 or newer")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "Clang too old! Use Clang 6.0 or newer")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(WARNING "Clang < 10.0 is not a official supported, please try a newer!")
    endif()
endif()


###############################################################################
# CPU Architecture: available instruction sets for e.g. SIMD extensions
#
# Conveniently set the architecture for the CPU compiler via this option.
# For unsupported compilers, ignore this option and set CXXFLAGS.
###############################################################################

set(PMACC_CPU_ARCH $ENV{PMACC_CPU_ARCH} CACHE STRING
    "compiler dependent CPU architecture string"
)

# list of known compiler flags to set the CPU architecture
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64le")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={} -mtune={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "riscv64")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if("${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-x{}")
    endif()
# Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "riscv64")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
# XL
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(PMACC_CPU_ARCH_TEMPLATE "-qarch={}")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    if(NOT "${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-tp={}")
    endif()
endif()

# architecture is set and compiler is known
if(PMACC_CPU_ARCH AND PMACC_CPU_ARCH_TEMPLATE)
    string(REPLACE
       "{}"
       "${PMACC_CPU_ARCH}"
       PMACC_CPU_ARCH_STRING
       "${PMACC_CPU_ARCH_TEMPLATE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PMACC_CPU_ARCH_STRING}")
endif()


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
target_link_libraries(pmacc PUBLIC MPI::MPI_CXX)
target_link_libraries(pmacc_filesystem INTERFACE MPI::MPI_CXX)

if( CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "STATIC_LIBRARY" AND CMAKE_EXE_LINKER_FLAGS)
    # Workaround for linker issues when linking static MPI libraries.
    # Because of CMAKE_TRY_COMPILE_TARGET_TYPE CMake is providing the statics libraries before the object file from
    # `add_executable` therefore MPI symbols can not be resolved. Linking the linker flaks to the target again will
    # workaround the issue.
    target_link_libraries(pmacc PUBLIC ${CMAKE_EXE_LINKER_FLAGS})
endif()


################################################################################
# Find Threads
################################################################################

if(NOT THREADS_PREFER_PTHREAD_FLAG)
     set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif()
find_package(Threads REQUIRED)
if(NOT APPLE)
    # librt: undefined reference to `clock_gettime'
    find_library(RT_LIBRARY rt)
    if(RT_LIBRARY)
        target_link_libraries(pmacc PUBLIC ${RT_LIBRARY})
    endif()
endif()


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.74 REQUIRED COMPONENTS program_options)
if(TARGET Boost::boost)
    target_link_libraries(pmacc PUBLIC Boost::boost)
    target_link_libraries(pmacc PUBLIC Boost::program_options)
else()
    target_include_directories(pmacc PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(pmacc PUBLIC ${Boost_LIBRARIES})
endif()

# We do not use std::auto_ptr and keeping this enabled in Boost causes a
# warning with NVCC+GCC and is unnecessary time spend in compile time
# (note that std::auto_ptr is deprecated in C++11 and removed in C++17)
message(STATUS "Boost: deactivate std::auto_ptr")
target_compile_definitions(pmacc PUBLIC "-DBOOST_NO_AUTO_PTR")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Boost: Disable variadic templates")
    message(STATUS "Boost: Do not use fenv.h from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # suppress boost error
    # 'no member named "impl" in "boost::detail::thread_move_t<boost::detail::nullary_function<void ()> >"'
    # in 'boost/thread/detail/nullary_function.hpp'
    message(STATUS "Boost: Do not use C++11 smart pointers from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SMART_PTR")
endif()

# Newer Boost releases: probably troublesome, warn at least
if(Boost_VERSION GREATER 107000)
    message(WARNING "Untested Boost release > 1.70.0 (Found ${Boost_VERSION})! "
                    "Maybe use a newer PIConGPU?")
endif()

################################################################################
# Find OpenMP
################################################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND (alpaka_ACC_GPU_HIP_ENABLE OR (alpaka_ACC_GPU_CUDA_ENABLE AND alpaka_CUDA_COMPILER MATCHES "clang")))
    # For HIP the problem is that in alpaka '::isnan(), ::sinh(), ::isfinite(), ::isinf()' is not found.
    # The reason could be that if OpenMP is activated clang is using math C headers where all of these functions are macros.
    message(WARNING "OpenMP host side acceleration is disabled: CUDA/HIP compilation with clang is not supporting OpenMP.")
else()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


################################################################################
# Find mallocMC
################################################################################

if(alpaka_ACC_GPU_CUDA_ENABLE OR alpaka_ACC_GPU_HIP_ENABLE)
    set(mallocMC_alpaka_PROVIDER "extern" CACHE STRING "Select which alpaka is used for mallocMC")
    find_package(mallocMC 2.6.0 QUIET)

    if(NOT mallocMC_FOUND)
        message(STATUS "Using mallocMC from thirdParty/ directory")
        set(MALLOCMC_ROOT "${PMacc_DIR}/../../thirdParty/mallocMC")
        find_package(mallocMC 2.6.0 REQUIRED)
    endif(NOT mallocMC_FOUND)

    target_include_directories(pmacc PUBLIC ${mallocMC_INCLUDE_DIRS})
    target_link_libraries(pmacc PUBLIC ${mallocMC_LIBRARIES})
    target_compile_definitions(pmacc PUBLIC ${mallocMC_DEFINITIONS})
endif()


################################################################################
# PMacc options
################################################################################

option(PMACC_ASYNC_QUEUES "Enable asynchronous alpaka queues" ON)
if(PMACC_ASYNC_QUEUES)
    target_compile_definitions(pmacc PUBLIC "PMACC_USE_ASYNC_QUEUES=1")
endif()
option(PMACC_BLOCKING_KERNEL
    "activate checks for every kernel call and synch after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    target_compile_definitions(pmacc PUBLIC "-DPMACC_SYNC_KERNEL=1")
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for PMacc")
target_compile_definitions(pmacc PUBLIC "-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE}")
