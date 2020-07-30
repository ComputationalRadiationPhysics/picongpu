#
# Copyright 2014-2020 Benjamin Worpitz, Erik Zenker, Axel Huebl, Jan Stephan
#                     Rene Widera
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

include(CMakePrintHelpers) # for easier printing of variables and properties

#-------------------------------------------------------------------------------
# Options.
set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE_DEFAULT ON)

set(ALPAKA_EMU_MEMCPY3D_DEFAULT OFF)

# HIP and platform selection and warning about unsupported features
option(ALPAKA_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)
option(ALPAKA_ACC_GPU_HIP_ONLY_MODE "Only back-ends using HIP can be enabled in this mode." OFF) # HIP only runs without other back-ends

# Drop-down combo box in cmake-gui for HIP platforms.
set(ALPAKA_HIP_PLATFORM "nvcc" CACHE STRING "Specify HIP platform")
set_property(CACHE ALPAKA_HIP_PLATFORM PROPERTY STRINGS "nvcc;clang")

if(ALPAKA_ACC_GPU_HIP_ENABLE AND NOT ALPAKA_ACC_GPU_HIP_ONLY_MODE AND ALPAKA_HIP_PLATFORM MATCHES "nvcc")
    message(WARNING "HIP back-end must be used together with ALPAKA_ACC_GPU_HIP_ONLY_MODE")
    set(ALPAKA_ACC_GPU_HIP_ENABLE OFF CACHE BOOL "" FORCE)
endif()

if(ALPAKA_ACC_GPU_HIP_ENABLE AND ALPAKA_HIP_PLATFORM MATCHES "clang")
    message(WARNING
        "The HIP back-end is currently experimental."
        "alpaka HIP backend compiled with clang does not support callback functions."
        )
endif()

option(ALPAKA_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode (This allows to mix alpaka code with native CUDA code)." OFF)
# If CUDA-only mode is enabled, we set the defaults for all CPU back-ends to OFF.
# If they are explicitly set via the command line, the user will get an error later on.
if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE OR ALPAKA_ACC_GPU_HIP_ONLY_MODE) # CUDA-only or HIP-only
    set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE_DEFAULT OFF)
    set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE_DEFAULT OFF)
endif()

option(ALPAKA_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" ON)

# If CUDA is enabled, we set the defaults for some unsupported back-ends to OFF.
# If they are explicitly set via the command line, the user will get an error later on.
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT OFF)
    if(ALPAKA_CUDA_COMPILER MATCHES "clang")
        set(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE_DEFAULT OFF)
        set(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE_DEFAULT OFF)
        set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE_DEFAULT OFF)
    endif()
endif()

if(ALPAKA_ACC_GPU_HIP_ENABLE)
    set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT OFF)
    set(ALPAKA_EMU_MEMCPY3D_DEFAULT ON)
endif()

if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE AND NOT ALPAKA_ACC_GPU_CUDA_ENABLE)
    message(WARNING "If ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled, ALPAKA_ACC_GPU_CUDA_ENABLE has to be enabled as well.")
    set(_ALPAKA_FOUND FALSE)
endif()
if(ALPAKA_ACC_GPU_HIP_ONLY_MODE AND NOT ALPAKA_ACC_GPU_HIP_ENABLE)
    message(WARNING "If ALPAKA_ACC_GPU_HIP_ONLY_MODE is enabled, ALPAKA_ACC_GPU_HIP_ENABLE has to be enabled as well.")
    set(_ALPAKA_FOUND FALSE)
endif()

option(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" ${ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" ${ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread back-end" ${ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" ${ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" ${ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" ${ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE_DEFAULT})
option(ALPAKA_ACC_CPU_BT_OMP4_ENABLE "Enable the OpenMP 4.0 CPU block and block thread back-end" ${ALPAKA_ACC_CPU_BT_OMP4_ENABLE_DEFAULT})

option(ALPAKA_EMU_MEMCPY3D "Emulate internal used hip/cuda-Memcpy3D(async) with a kernel" ${ALPAKA_EMU_MEMCPY3D_DEFAULT})

if((ALPAKA_ACC_GPU_CUDA_ONLY_MODE OR ALPAKA_ACC_GPU_HIP_ONLY_MODE)
   AND
    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OR
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
    ALPAKA_ACC_CPU_BT_OMP4_ENABLE))
    if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
        message(WARNING "If ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled, only back-ends using CUDA can be enabled! This allows to mix alpaka code with native CUDA code. However, this prevents any non-CUDA back-ends from being enabled.")
    endif()
    if(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
        message(WARNING "If ALPAKA_ACC_GPU_HIP_ONLY_MODE is enabled, only back-ends using HIP can be enabled!")
    endif()
    set(_ALPAKA_FOUND FALSE)
endif()

# avoids CUDA+HIP conflict
if(ALPAKA_ACC_GPU_HIP_ENABLE AND ALPAKA_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "CUDA and HIP can not be enabled both at the same time.")
endif()

# HIP is only supported on Linux
if(ALPAKA_ACC_GPU_HIP_ENABLE AND (MSVC OR WIN32))
    message(WARNING "Optional alpaka dependency HIP can not be built on Windows! HIP back-end disabled!")
    set(ALPAKA_ACC_GPU_HIP_ENABLE OFF CACHE BOOL "Enable the HIP GPU back-end" FORCE)
endif()

# Drop-down combo box in cmake-gui.
set(ALPAKA_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE ALPAKA_DEBUG PROPERTY STRINGS "0;1;2")

set(ALPAKA_CXX_STANDARD "14" CACHE STRING "C++ standard version")
set_property(CACHE ALPAKA_CXX_STANDARD PROPERTY STRINGS "14;17;20")

if(NOT TARGET alpaka)
    add_library(alpaka INTERFACE)
    
    target_compile_features(alpaka INTERFACE cxx_std_${ALPAKA_CXX_STANDARD})

    add_library(alpaka::alpaka ALIAS alpaka)
endif()

option(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST "Allow host-only contructs like assert in offload code in debug mode." ON)
set(ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB "30" CACHE STRING "Kibibytes (1024B) of memory to allocate for block shared memory for backends requiring static allocation (includes CPU_B_OMP2_T_SEQ, CPU_B_TBB_T_SEQ, CPU_B_SEQ_T_SEQ)")

#-------------------------------------------------------------------------------
# Debug output of common variables.
if(${ALPAKA_DEBUG} GREATER 1)
    cmake_print_variables(_ALPAKA_ROOT_DIR)
    cmake_print_variables(_ALPAKA_COMMON_FILE)
    cmake_print_variables(_ALPAKA_ADD_EXECUTABLE_FILE)
    cmake_print_variables(_ALPAKA_ADD_LIBRARY_FILE)
    cmake_print_variables(CMAKE_BUILD_TYPE)
endif()

#-------------------------------------------------------------------------------
# Check supported compilers.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
    message(FATAL_ERROR "Clang versions < 4.0 are not supported!")
endif()

if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE AND (ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE))
    message(FATAL_ERROR "Fibers and CUDA or HIP back-end can not be enabled both at the same time.")
endif()

#-------------------------------------------------------------------------------
# Compiler settings.

if(MSVC)
    # CUDA\v9.2\include\crt/host_runtime.h(265): warning C4505: '__cudaUnregisterBinaryUtil': unreferenced local function has been removed
    if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
        target_compile_options(alpaka INTERFACE "/wd4505")
    endif()
else()
    find_package(Threads REQUIRED)
    target_link_libraries(alpaka INTERFACE Threads::Threads)

    if(NOT APPLE)
        # librt: undefined reference to `clock_gettime'
        find_library(RT_LIBRARY rt)
        if(RT_LIBRARY)
            target_link_libraries(alpaka INTERFACE ${RT_LIBRARY})
        endif()
    endif()
endif()

#-------------------------------------------------------------------------------
# Find Boost.
set(_ALPAKA_BOOST_MIN_VER "1.65.1")

if(${ALPAKA_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
endif()

find_package(Boost ${_ALPAKA_BOOST_MIN_VER} REQUIRED
             OPTIONAL_COMPONENTS fiber)

target_link_libraries(alpaka INTERFACE Boost::headers)

if(NOT Boost_FIBER_FOUND)
    message(STATUS "Optional alpaka dependency Boost.Fiber could not be found! Fiber back-end disabled!")
    set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OFF CACHE BOOL "Enable the Boost.Fiber CPU back-end" FORCE)
endif()

if(${ALPAKA_DEBUG} GREATER 1)
    message(STATUS "Boost in:")
    cmake_print_variables(BOOST_ROOT)
    cmake_print_variables(BOOSTROOT)
    cmake_print_variables(BOOST_INCLUDEDIR)
    cmake_print_variables(BOOST_LIBRARYDIR)
    cmake_print_variables(Boost_NO_SYSTEM_PATHS)
    cmake_print_variables(Boost_ADDITIONAL_VERSIONS)
    cmake_print_variables(Boost_USE_MULTITHREADED)
    cmake_print_variables(Boost_USE_STATIC_LIBS)
    cmake_print_variables(Boost_USE_STATIC_RUNTIME)
    cmake_print_variables(Boost_USE_DEBUG_RUNTIME)
    cmake_print_variables(Boost_USE_DEBUG_PYTHON)
    cmake_print_variables(Boost_USE_STLPORT)
    cmake_print_variables(Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS)
    cmake_print_variables(Boost_COMPILER)
    cmake_print_variables(Boost_THREADAPI)
    cmake_print_variables(Boost_NAMESPACE)
    cmake_print_variables(Boost_DEBUG)
    cmake_print_variables(Boost_DETAILED_FAILURE_MSG)
    cmake_print_variables(Boost_REALPATH)
    cmake_print_variables(Boost_NO_BOOST_CMAKE)
    message(STATUS "Boost out:")
    cmake_print_variables(Boost_FOUND)
    cmake_print_variables(Boost_INCLUDE_DIRS)
    cmake_print_variables(Boost_LIBRARY_DIRS)
    cmake_print_variables(Boost_LIBRARIES)
    cmake_print_variables(Boost_FIBER_FOUND)
    cmake_print_variables(Boost_FIBER_LIBRARY)
    cmake_print_variables(Boost_CONTEXT_FOUND)
    cmake_print_variables(Boost_CONTEXT_LIBRARY)
    cmake_print_variables(Boost_SYSTEM_FOUND)
    cmake_print_variables(Boost_SYSTEM_LIBRARY)
    cmake_print_variables(Boost_THREAD_FOUND)
    cmake_print_variables(Boost_THREAD_LIBRARY)
    cmake_print_variables(Boost_ATOMIC_FOUND)
    cmake_print_variables(Boost_ATOMIC_LIBRARY)
    cmake_print_variables(Boost_CHRONO_FOUND)
    cmake_print_variables(Boost_CHRONO_LIBRARY)
    cmake_print_variables(Boost_DATE_TIME_FOUND)
    cmake_print_variables(Boost_DATE_TIME_LIBRARY)
    cmake_print_variables(Boost_VERSION)
    cmake_print_variables(Boost_LIB_VERSION)
    cmake_print_variables(Boost_MAJOR_VERSION)
    cmake_print_variables(Boost_MINOR_VERSION)
    cmake_print_variables(Boost_SUBMINOR_VERSION)
    cmake_print_variables(Boost_LIB_DIAGNOSTIC_DEFINITIONS)
    message(STATUS "Boost cached:")
    cmake_print_variables(Boost_INCLUDE_DIR)
    cmake_print_variables(Boost_LIBRARY_DIR)
endif()

#-------------------------------------------------------------------------------
# Find TBB.
if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(alpaka INTERFACE TBB::tbb)
    else()
        message(STATUS "Optional alpaka dependency TBB could not be found! TBB grid block back-end disabled!")
        set(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE OFF CACHE BOOL "Enable the TBB grid block back-end" FORCE)
    endif()
endif()

#-------------------------------------------------------------------------------
# Find OpenMP.
if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        if(OpenMP_CXX_VERSION VERSION_LESS 4.0)
            set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE OFF CACHE BOOL "Enable the OpenMP 4.0 CPU block and thread back-end" FORCE)
        endif()

        target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)

        # Clang versions starting from 3.9 support OpenMP 4.0 only when given the corresponding flag
        if(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
            target_link_options(alpaka INTERFACE $<$<CXX_COMPILER_ID:AppleClang,Clang>:-fopenmp-version=40>)
        endif()
    else()
        message(STATUS "Optional alpaka dependency OpenMP could not be found! OpenMP back-ends disabled!")
        set(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU grid block back-end" FORCE)
        set(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU block thread back-end" FORCE)
        set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE OFF CACHE BOOL "Enable the OpenMP 4.0 CPU block and thread back-end" FORCE)
    endif()
endif()

#-------------------------------------------------------------------------------
# Find CUDA.
if(ALPAKA_ACC_GPU_CUDA_ENABLE)

    if(NOT DEFINED ALPAKA_CUDA_VERSION)
        set(ALPAKA_CUDA_VERSION 9.0)
    endif()

    if(ALPAKA_CUDA_VERSION VERSION_LESS 9.0)
        message(WARNING "CUDA Toolkit < 9.0 is not supported!")
        set(_ALPAKA_FOUND FALSE)

    else()
        find_package(CUDA "${ALPAKA_CUDA_VERSION}")
        if(NOT CUDA_FOUND)
            message(STATUS "Optional alpaka dependency CUDA could not be found! CUDA back-end disabled!")
            set(ALPAKA_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA GPU back-end" FORCE)
        else()
            set(ALPAKA_CUDA_VERSION "${CUDA_VERSION}")
            if(CUDA_VERSION VERSION_LESS 10.3)
                set(ALPAKA_CUDA_ARCH "30" CACHE STRING "GPU architecture")
            else()
                set(ALPAKA_CUDA_ARCH "35" CACHE STRING "GPU architecture")
            endif()
            set(ALPAKA_CUDA_COMPILER "nvcc" CACHE STRING "CUDA compiler")
            set_property(CACHE ALPAKA_CUDA_COMPILER PROPERTY STRINGS "nvcc;clang")

            option(ALPAKA_CUDA_FAST_MATH "Enable fast-math" ON)
            option(ALPAKA_CUDA_FTZ "Set flush to zero for GPU" OFF)
            option(ALPAKA_CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
            option(ALPAKA_CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
            option(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA "Enable experimental, extended host-device lambdas in NVCC" ON)
            option(ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION "Enable separable compilation in NVCC" OFF)

            if(ALPAKA_CUDA_COMPILER MATCHES "clang")
                if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                    message(FATAL_ERROR "Using clang as CUDA compiler is only possible if clang is the host compiler!")
                endif()

                if(CMAKE_CXX_COMPILER_VERSION LESS 6.0)
                    if(CUDA_VERSION GREATER_EQUAL 9.0)
                        message(FATAL_ERROR "Clang versions lower than 6 do not support CUDA 9 or greater!")
                    endif()
                elseif(CMAKE_CXX_COMPILER_VERSION LESS 7.0)
                    if(CUDA_VERSION GREATER_EQUAL 9.1)
                        message(FATAL_ERROR "Clang versions lower than 7 do not support CUDA 9.1 or greater!")
                    endif()
                elseif(CMAKE_CXX_COMPILER_VERSION LESS 8.0)
                    if(CUDA_VERSION GREATER_EQUAL 10.0)
                        message(FATAL_ERROR "Clang versions lower than 8 do not support CUDA 10.0 or greater!")
                    endif()
                elseif(CMAKE_CXX_COMPILER_VERSION LESS 9.0)
                    if(CUDA_VERSION GREATER_EQUAL 10.1)
                        message(FATAL_ERROR "Clang versions lower than 9 do not support CUDA 10.1 or greater!")
                    endif()
                elseif(CMAKE_CXX_COMPILER_VERSION LESS 10.0)
                    if(CUDA_VERSION GREATER_EQUAL 10.2)
                        message(FATAL_ERROR "Clang versions lower than 10 do not support CUDA 10.2 or greater!")
                    endif()
                endif()

                if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
                    message(FATAL_ERROR "Clang as a CUDA compiler does not support boost.fiber!")
                endif()
                if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
                    message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 2!")
                endif()
                if(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
                    message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 4!")
                endif()

                foreach(_CUDA_ARCH_ELEM ${ALPAKA_CUDA_ARCH})
                    target_compile_options(alpaka INTERFACE  "--cuda-gpu-arch=sm_${_CUDA_ARCH_ELEM}")
                endforeach()

                target_compile_options(alpaka INTERFACE "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

                # This flag silences the warning produced by the Dummy.cpp files:
                # clang: warning: argument unused during compilation: '--cuda-gpu-arch=sm_XX'
                # This seems to be a false positive as all flags are 'unused' for an empty file.
                target_compile_options(alpaka INTERFACE "-Qunused-arguments")

                # Silences warnings that are produced by boost because clang is not correctly identified.
                target_compile_options(alpaka INTERFACE "-Wno-unused-local-typedef")

                if(ALPAKA_CUDA_FAST_MATH)
                    # -ffp-contract=fast enables the usage of FMA
                    target_compile_options(alpaka INTERFACE "-ffast-math" "-ffp-contract=fast")
                endif()

                if(ALPAKA_CUDA_FTZ)
                    target_compile_options(alpaka INTERFACE "-fcuda-flush-denormals-to-zero")
                endif()

                if(ALPAKA_CUDA_SHOW_REGISTER)
                    target_compile_options(alpaka INTERFACE "-Xcuda-ptxas=-v")
                endif()

                if(ALPAKA_CUDA_KEEP_FILES)
                    target_compile_options(alpaka INTERFACE "-save-temps")
                endif()

                # When libstdc++ is used and -std=gnu++XX is set, we get the following compile error:
                # /usr/lib/gcc/x86_64-linux-gnu/5.5.0/../../../../include/c++/5.5.0/type_traits:311:39: error: __float128 is not supported on this target struct __is_floating_point_helper<__float128>
                # Clang doesn't support the __float128 type (at least when building CUDA device code)
                # * Due to the minimum requirement to compile with C++14 and because extensions are enabled by default by CMake, it adds -std=gnu++14 instead of -std=c++14 to the command line.
                #   Due to alpaka being an INTERFACE library (header-only) we are not allowed to set CXX_EXTENSIONS to OFF and transitively disable extensions for inherited targets.
                # * Defining __float128 on the command line is the least invasive workaround found here: https://bugs.llvm.org/show_bug.cgi?id=13530#c6
                target_compile_definitions(alpaka INTERFACE "__float128=void")

                # CMake 3.15 does not provide the `--std=c++*` argument to clang anymore.
                # It is not necessary for basic c++ compilation because clangs default is already higher, but CUDA code compiled with -x cuda still defaults to c++98.
                if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15.0")
                    target_compile_options(alpaka INTERFACE "-std=c++${ALPAKA_CXX_STANDARD}")
                endif()

            else()
                if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                    if((CUDA_VERSION VERSION_EQUAL 9.0) OR (CUDA_VERSION VERSION_EQUAL 9.1))
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 6.0)
                            message(FATAL_ERROR "NVCC 9.0 - 9.1 do not support GCC 7+ and fail compiling the std::tuple implementation in GCC 6+. Please use GCC 5!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 9.2)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 8.0)
                            message(FATAL_ERROR "NVCC 9.2 does not support GCC 8+. Please use GCC 5, 6 or 7!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.0)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 8.0)
                            message(FATAL_ERROR "NVCC 10.0 does not support GCC 8+. Please use GCC 5, 6 or 7!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.1)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 9.0)
                            message(FATAL_ERROR "NVCC 10.1 does not support GCC 9+. Please use GCC 5, 6, 7 or 8!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.2)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 9.0)
                            message(FATAL_ERROR "NVCC 10.2 does not support GCC 9+. Please use GCC 5, 6, 7 or 8!")
                        endif()
                    endif()
                elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                    if(CUDA_VERSION VERSION_EQUAL 9.0)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 4.0)
                            message(FATAL_ERROR "NVCC 9.0 does not support clang 4+. Please use NVCC 9.1!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 9.1)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 5.0)
                            message(FATAL_ERROR "NVCC 9.1 does not support clang 5+. Please use clang 4!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 9.2)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 5.0)
                            message(FATAL_ERROR "NVCC 9.2 does not support clang 6+ and fails compiling with clang 5. Please use clang 4!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.0)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 7.0)
                            message(FATAL_ERROR "NVCC 10.0 does not support clang 7+. Please use clang 4, 5 or 6!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.1)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 9.0)
                            message(FATAL_ERROR "NVCC 10.1 does not support clang 9+. Please use clang 4, 5, 6, 7 or 8!")
                        endif()
                    elseif(CUDA_VERSION VERSION_EQUAL 10.2)
                        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 9.0)
                            message(FATAL_ERROR "NVCC 10.2 does not support clang 9+. Please use clang 4, 5, 6, 7 or 8!")
                        endif()
                    endif()
                endif()

                if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
                    message(FATAL_ERROR "NVCC does not support boost.fiber!")
                endif()

                # Clean up the flags. Else, multiple find calls would result in duplicate flags. Furthermore, other modules may have set different settings.
                set(CUDA_NVCC_FLAGS)

                if(${ALPAKA_DEBUG} GREATER 1)
                    set(CUDA_VERBOSE_BUILD ON)
                endif()

                set(CUDA_PROPAGATE_HOST_FLAGS ON)

                if(ALPAKA_CUDA_NVCC_SEPARABLE_COMPILATION)
                    set(CUDA_SEPARABLE_COMPILATION ON)
                endif()

                # nvcc sets no linux/__linux macros on OpenPOWER linux
                # nvidia bug id: 2448610
                if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
                        list(APPEND CUDA_NVCC_FLAGS -Dlinux)
                    endif()
                endif()

                # NOTE: Since CUDA 10.2 this option is also alternatively called '--extended-lambda'
                if(ALPAKA_CUDA_NVCC_EXPT_EXTENDED_LAMBDA)
                    list(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)
                endif()
                # This is mandatory because with c++14 many standard library functions we rely on are constexpr (std::min, std::multiplies, ...)
                list(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)

                foreach(_CUDA_ARCH_ELEM ${ALPAKA_CUDA_ARCH})
                    # set flags to create device code for the given architecture
                    list(APPEND CUDA_NVCC_FLAGS
                        --generate-code=arch=compute_${_CUDA_ARCH_ELEM},code=sm_${_CUDA_ARCH_ELEM}
                        --generate-code=arch=compute_${_CUDA_ARCH_ELEM},code=compute_${_CUDA_ARCH_ELEM}
                    )
                endforeach()

                if(NOT MSVC)
                    list(APPEND CUDA_NVCC_FLAGS -std=c++${ALPAKA_CXX_STANDARD})
                endif()

                set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

                if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
                    list(APPEND CUDA_NVCC_FLAGS -g)
                    if(MSVC)
                        message(WARNING "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not support -G with CUDA! "
                                        "Device debug symbols NOT added.")
                    else()
                        list(APPEND CUDA_NVCC_FLAGS -G)
                    endif()
                endif()

                if(ALPAKA_CUDA_FAST_MATH)
                    list(APPEND CUDA_NVCC_FLAGS --use_fast_math)
                endif()

                if(ALPAKA_CUDA_FTZ)
                    list(APPEND CUDA_NVCC_FLAGS --ftz=true)
                else()
                    list(APPEND CUDA_NVCC_FLAGS --ftz=false)
                endif()

                if(ALPAKA_CUDA_SHOW_REGISTER)
                    list(APPEND CUDA_NVCC_FLAGS -Xptxas=-v)
                endif()

                # Always add warning/error numbers which can be used for suppressions
                list(APPEND CUDA_NVCC_FLAGS -Xcudafe=--display_error_number)

                # avoids warnings on host-device signatured, default constructors/destructors
                list(APPEND CUDA_NVCC_FLAGS -Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored)

                # avoids warnings on host-device signature of 'std::__shared_count<>'
                if(CUDA_VERSION EQUAL 10.0)
                    list(APPEND CUDA_NVCC_FLAGS -Xcudafe=--diag_suppress=2905)
                elseif(CUDA_VERSION EQUAL 10.1)
                    list(APPEND CUDA_NVCC_FLAGS -Xcudafe=--diag_suppress=2912)
                elseif(CUDA_VERSION EQUAL 10.2)
                    list(APPEND CUDA_NVCC_FLAGS -Xcudafe=--diag_suppress=2976)
                endif()

                if(ALPAKA_CUDA_KEEP_FILES)
                    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/nvcc_tmp")
                    list(APPEND CUDA_NVCC_FLAGS --keep)
                    list(APPEND CUDA_NVCC_FLAGS --keep-dir="${PROJECT_BINARY_DIR}/nvcc_tmp")
                endif()

                option(ALPAKA_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
                if(ALPAKA_CUDA_SHOW_CODELINES)
                    list(APPEND CUDA_NVCC_FLAGS --source-in-ptx -lineinfo)
                    if(NOT MSVC)
                        list(APPEND CUDA_NVCC_FLAGS -Xcompiler=-rdynamic)
                    endif()
                    set(ALPAKA_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
                endif()
            endif()

            if(OpenMP_CXX_FOUND)
                # correctly propagate OpenMP flags
                # This can be removed once we support CMake's first class CUDA support.
                target_compile_options(alpaka INTERFACE ${OpenMP_CXX_FLAGS})
            endif()

            target_link_libraries(alpaka INTERFACE ${CUDA_CUDART_LIBRARY})
            target_include_directories(alpaka INTERFACE ${CUDA_INCLUDE_DIRS})
        endif()
    endif()
endif()

#-------------------------------------------------------------------------------
# Find HIP.
if(ALPAKA_ACC_GPU_HIP_ENABLE)

    if(NOT DEFINED ALPAKA_HIP_VERSION)
        set(ALPAKA_HIP_VERSION 3.3)
    endif()

    if(ALPAKA_HIP_VERSION VERSION_LESS 3.3)
        message(WARNING "HIP < 3.3 is not supported!")
        set(_ALPAKA_FOUND FALSE)
    else()
        # must set this for HIP package (note that you also need certain env vars)
        set(HIP_PLATFORM "${ALPAKA_HIP_PLATFORM}" CACHE STRING "")
        set(HIP_RUNTIME "${ALPAKA_HIP_PLATFORM}" CACHE STRING "")

        find_package(HIP "${ALPAKA_HIP_VERSION}")
        if(NOT HIP_FOUND)
            message(WARNING "Optional alpaka dependency HIP could not be found! HIP back-end disabled!")
            set(ALPAKA_ACC_GPU_HIP_ENABLE OFF CACHE BOOL "Enable the HIP GPU back-end" FORCE)
        else()
            set(ALPAKA_HIP_VERSION "${HIP_VERSION}")
            set(ALPAKA_HIP_COMPILER "hipcc" CACHE STRING "HIP compiler")
            set_property(CACHE ALPAKA_HIP_COMPILER PROPERTY STRINGS "hipcc")

            option(ALPAKA_HIP_FAST_MATH "Enable fast-math" ON)
            option(ALPAKA_HIP_FTZ "Set flush to zero for GPU" OFF)
            option(ALPAKA_HIP_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
            option(ALPAKA_HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)

            set(HIP_HIPCC_FLAGS)

            if(ALPAKA_HIP_PLATFORM MATCHES "nvcc")
                find_package(CUDA)
                if(NOT CUDA_FOUND)
                    message(WARNING "Could not find CUDA while HIP platform is set to nvcc. Compilation might fail.")
                endif()

                set(ALPAKA_CUDA_ARCH "30" CACHE STRING "GPU architecture")

                if(CUDA_VERSION VERSION_LESS 9.0)
                    message(WARNING "CUDA Toolkit < 9.0 is not supported!")
                    set(_ALPAKA_FOUND FALSE)
                endif()

                if(${ALPAKA_DEBUG} GREATER 1)
                    set(HIP_VERBOSE_BUILD ON)
                endif()

                list(APPEND HIP_NVCC_FLAGS --expt-extended-lambda)
                list(APPEND HIP_NVCC_FLAGS --expt-relaxed-constexpr)
                list(APPEND _ALPAKA_HIP_LIBRARIES "cudart")

                foreach(_HIP_ARCH_ELEM ${ALPAKA_CUDA_ARCH})
                    # set flags to create device code for the given architecture
                    list(APPEND CUDA_NVCC_FLAGS
                        --generate-code=arch=compute_${_HIP_ARCH_ELEM},code=sm_${_HIP_ARCH_ELEM}
                        --generate-code=arch=compute_${_HIP_ARCH_ELEM},code=compute_${_HIP_ARCH_ELEM}
                    )
                endforeach()
                # for CUDA cmake automatically adds compiler flags as nvcc does not do this,
                # but for HIP we have to do this here
                list(APPEND HIP_NVCC_FLAGS -D__CUDACC__)
                list(APPEND HIP_NVCC_FLAGS -ccbin ${CMAKE_CXX_COMPILER})
                list(APPEND HIP_NVCC_FLAGS -Xcompiler=-g)

                if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
                    list(APPEND HIP_NVCC_FLAGS -G)
                endif()
                # propage host flags
                # SET(CUDA_PROPAGATE_HOST_FLAGS ON) # does not exist in HIP, so do it manually
                string(TOUPPER "${CMAKE_BUILD_TYPE}" build_config)
                foreach( _flag ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_config}})
                    list(APPEND HIP_NVCC_FLAGS -Xcompiler=${_flag})
                endforeach()

                if(ALPAKA_HIP_FAST_MATH)
                    list(APPEND HIP_HIPCC_FLAGS --use_fast_math)
                endif()

                if(ALPAKA_HIP_FTZ)
                    list(APPEND HIP_HIPCC_FLAGS --ftz=true)
                else()
                    list(APPEND HIP_HIPCC_FLAGS --ftz=false)
                endif()

                if(ALPAKA_HIP_SHOW_REGISTER)
                    list(APPEND HIP_HIPCC_FLAGS -Xptxas=-v)
                endif()

                # avoids warnings on host-device signatured, default constructors/destructors
                list(APPEND HIP_HIPCC_FLAGS -Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored)

                # random numbers library ( HIP(NVCC) ) /hiprand
                # HIP_ROOT_DIR is set by FindHIP.cmake
                find_path(HIP_RAND_INC
                    NAMES "hiprand_kernel.h"
                    PATHS "${HIP_ROOT_DIR}/hiprand" "${HIP_ROOT_DIR}/include" "hiprand"
                    PATHS "/opt/rocm/rocrand/hiprand"
                    PATH_SUFFIXES "include" "hiprand")
                find_library(HIP_RAND_LIBRARY
                    NAMES "hiprand-d" "hiprand"
                    PATHS "${HIP_ROOT_DIR}/hiprand" "${HIP_ROOT_DIR}" "hiprand"
                    PATHS "/opt/rocm/rocrand/hiprand"
                    ENV HIP_PATH
                    PATH_SUFFIXES "lib" "lib64")
                if(NOT HIP_RAND_INC)
                    message(FATAL_ERROR "Could not find hipRAND include (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}).")
                endif()
                if(NOT HIP_RAND_LIBRARY)
                    message(FATAL_ERROR "Could not find hipRAND library (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}).")
                endif()
                target_include_directories(alpaka INTERFACE ${HIP_RAND_INC})
                target_link_libraries(alpaka INTERFACE ${HIP_RAND_LIBRARY})
            elseif(ALPAKA_HIP_PLATFORM MATCHES "clang")
                # # hiprand requires ROCm implementation of random numbers by rocrand
                FIND_PACKAGE(rocrand REQUIRED CONFIG                 
                    HINTS "${HIP_ROOT_DIR}/rocrand"
                    HINTS "/opt/rocm/rocrand")
                IF(rocrand_FOUND)
                    target_include_directories(alpaka INTERFACE ${rocrand_INCLUDE_DIRS})
                    # ATTENTION: rocRand libraries are not required by alpaka
                ELSE()
                    MESSAGE(FATAL_ERROR "Could not find rocRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/rocrand).")
                ENDIF()
            endif() 

            # # HIP random numbers
            FIND_PACKAGE(hiprand REQUIRED CONFIG 
                HINTS "${HIP_ROOT_DIR}/hiprand"
                HINTS "/opt/rocm/hiprand")
            IF(hiprand_FOUND)
                target_include_directories(alpaka INTERFACE ${hiprand_INCLUDE_DIRS})
                # ATTENTION: hipRand libraries are not required by alpaka
            ELSE()
                MESSAGE(FATAL_ERROR "Could not find hipRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/hiprand).")
            ENDIF()

            list(APPEND HIP_HIPCC_FLAGS -D__HIPCC__)
            list(APPEND HIP_HIPCC_FLAGS -std=c++${ALPAKA_CXX_STANDARD})

            if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
                list(APPEND HIP_HIPCC_FLAGS -g)
            endif()

            if(ALPAKA_HIP_KEEP_FILES)
                file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/hip_tmp")
                list(APPEND HIP_HIPCC_FLAGS --keep)
                list(APPEND HIP_HIPCC_FLAGS --keep-dir "${PROJECT_BINARY_DIR}/hip_tmp")
            endif()

            option(ALPAKA_HIP_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
            if(ALPAKA_HIP_SHOW_CODELINES)
                list(APPEND HIP_HIPCC_FLAGS --source-in-ptx -lineinfo)
                list(APPEND HIP_HIPCC_FLAGS -Xcompiler=rdynamic)
                set(ALPAKA_HIP_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
            endif()
            if(_ALPAKA_HIP_LIBRARIES)
                target_link_libraries(alpaka INTERFACE ${_ALPAKA_HIP_LIBRARIES})
            endif()
        endif()
    endif()
endif() # HIP

#-------------------------------------------------------------------------------
# alpaka.
if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ONLY_MODE")
    message(STATUS ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
endif()

if(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ONLY_MODE")
    message(STATUS ALPAKA_ACC_GPU_HIP_ONLY_MODE)
endif()

if(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()

if(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED")

    if(MSVC AND (${CMAKE_SIZEOF_VOID_P} EQUAL 4))
        # On Win32 boost context triggers:
        # libboost_context-vc141-mt-gd-1_64.lib(jump_i386_ms_pe_masm.obj) : error LNK2026: module unsafe for SAFESEH image.
        target_link_options(Boost::fiber INTERFACE "/SAFESEH:NO")
    endif()
    target_link_libraries(alpaka INTERFACE Boost::fiber)

    message(STATUS ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()
if(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
endif()
if(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_BT_OMP4_ENABLED")
    message(STATUS ALPAKA_ACC_CPU_BT_OMP4_ENABLED)
endif()
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ENABLED")
    message(STATUS ALPAKA_ACC_GPU_CUDA_ENABLED)
endif()
if(ALPAKA_ACC_GPU_HIP_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ENABLED")
    message(STATUS ALPAKA_ACC_GPU_HIP_ENABLED)
endif()

if(ALPAKA_EMU_MEMCPY3D)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_EMU_MEMCPY3D_ENABLED")
endif()

target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
if(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST)
   target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST")
endif()
target_compile_definitions(alpaka INTERFACE "ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=${ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB}")

if(ALPAKA_CI)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_CI")
endif()

# cxx flags will not be forwarded to hip wrapped compiler, so it has to be provided manually
if(ALPAKA_ACC_GPU_HIP_ENABLE)
    get_property(_ALPAKA_COMPILE_DEFINITIONS_HIP
                 TARGET alpaka
                 PROPERTY INTERFACE_COMPILE_DEFINITIONS)
    list_add_prefix("-D" _ALPAKA_COMPILE_DEFINITIONS_HIP)
    list(APPEND HIP_HIPCC_FLAGS
        ${_ALPAKA_COMPILE_DEFINITIONS_HIP}
        )
    HIP_INCLUDE_DIRECTORIES(
        # ${_ALPAKA_INCLUDE_DIRECTORY}
        # ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC}
        ${HIP_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${_ALPAKA_ROOT_DIR}/test/common/include
        )

    if(OpenMP_CXX_FOUND) # remove fopenmp link from nvcc, otherwise linker error will occur
        get_property(_ALPAKA_LINK_LIBRARIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_LIBRARIES)
        list(REMOVE_ITEM _ALPAKA_LINK_LIBRARIES_PUBLIC "OpenMP::OpenMP_CXX")

        target_link_options(alpaka INTERFACE "-Xcompiler ${OpenMP_CXX_FLAGS}")
        set_property(TARGET alpaka
                     PROPERTY INTERFACE_LINK_LIBRARIES ${_ALPAKA_LINK_LIBRARIES_PUBLIC})
    endif()
    if(ALPAKA_HIP_PLATFORM MATCHES "clang")
        # GFX600, GFX601, GFX700, GFX701, GFX702, GFX703, GFX704, GFX801, GFX802, GFX803, GFX810, GFX900, GFX902
        target_link_options(alpaka INTERFACE "--amdgpu-target=gfx803")
        target_link_options(alpaka INTERFACE "--amdgpu-target=gfx900")
        target_link_options(alpaka INTERFACE "--amdgpu-target=gfx906")
    endif()
endif()

#-------------------------------------------------------------------------------
# Target.
if(TARGET alpaka)

    if(${ALPAKA_DEBUG} GREATER 1)
        # Compile options.
        get_property(_ALPAKA_COMPILE_OPTIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_OPTIONS)
        cmake_print_variables(_ALPAKA_COMPILE_OPTIONS_PUBLIC)

        # Compile definitions
        get_property(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_DEFINITIONS)
        cmake_print_variables(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC)

        # Include directories.
        get_property(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        cmake_print_variables(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC)
    endif()

    # the alpaka library itself
    target_include_directories(alpaka INTERFACE ${_ALPAKA_INCLUDE_DIRECTORY})

    if(${ALPAKA_DEBUG} GREATER 1)
        # Link libraries.
        # There are no PUBLIC_LINK_FLAGS in CMAKE:
        # http://stackoverflow.com/questions/26850889/cmake-keeping-link-flags-of-internal-libs
        get_property(_ALPAKA_LINK_LIBRARIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_LIBRARIES)
        cmake_print_variables(_ALPAKA_LINK_LIBRARIES_PUBLIC)

        get_property(_ALPAKA_LINK_FLAGS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_OPTIONS)
        cmake_print_variables(_ALPAKA_LINK_FLAGS_PUBLIC)
    endif()
endif()

# NVCC does not incorporate the COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
if((ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE) AND ALPAKA_CUDA_COMPILER MATCHES "nvcc")
    get_property(_ALPAKA_COMPILE_OPTIONS_PUBLIC
                 TARGET alpaka
                 PROPERTY INTERFACE_COMPILE_OPTIONS)
    string(REPLACE ";" " " _ALPAKA_COMPILE_OPTIONS_STRING "${_ALPAKA_COMPILE_OPTIONS_PUBLIC}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_ALPAKA_COMPILE_OPTIONS_STRING}")

    # Append CMAKE_CXX_FLAGS_[Release|Debug|RelWithDebInfo] to CMAKE_CXX_FLAGS
    # because FindCUDA only propagates the latter to nvcc.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" build_config)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_config}}")
endif()

