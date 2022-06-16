#
# Copyright 2022 Benjamin Worpitz, Erik Zenker, Axel Huebl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci, Bernhard Manfred Gruber
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

include(CMakePrintHelpers) # for easier printing of variables and properties
include(CMakeDependentOption) # Make options depend on other options

#-------------------------------------------------------------------------------
# Options.

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED alpaka_${name})
        set(alpaka_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE alpaka_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# alpaka_COMPILER_OPTIONS_HOST, alpaka_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'.")
    endif()
    if(type STREQUAL "list")
        set(${name} ${${name}} ${ARGN} PARENT_SCOPE)
    elseif(type STREQUAL "var")
        foreach(arg IN LISTS ARGN)
            set(tmp "${tmp} ${arg}")
        endforeach()
        set(${name} "${${name}} ${tmp}" PARENT_SCOPE)
    elseif(type STREQUAL "target")
        foreach(arg IN LISTS ARGN)
            target_compile_options(${name} INTERFACE ${arg})
        endforeach()
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'.")
    endif()
endfunction()

# HIP and platform selection and warning about unsupported features
option(alpaka_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)
option(alpaka_ACC_GPU_HIP_ONLY_MODE "Only back-ends using HIP can be enabled in this mode." OFF) # HIP only runs without other back-ends

option(alpaka_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" OFF)
option(alpaka_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode (This allows to mix alpaka code with native CUDA code)." OFF)

if(alpaka_ACC_GPU_CUDA_ONLY_MODE AND NOT alpaka_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "If alpaka_ACC_GPU_CUDA_ONLY_MODE is enabled, alpaka_ACC_GPU_CUDA_ENABLE has to be enabled as well.")
endif()
if(alpaka_ACC_GPU_HIP_ONLY_MODE AND NOT alpaka_ACC_GPU_HIP_ENABLE)
    message(FATAL_ERROR "If alpaka_ACC_GPU_HIP_ONLY_MODE is enabled, alpaka_ACC_GPU_HIP_ENABLE has to be enabled as well.")
endif()

option(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" OFF)
option(alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" OFF)
option(alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread back-end" OFF)
option(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" OFF)
option(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" OFF)
option(alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" OFF)
option(alpaka_ACC_ANY_BT_OMP5_ENABLE "Enable the OpenMP 5.0 CPU block and block thread back-end" OFF)
option(alpaka_ACC_ANY_BT_OACC_ENABLE "Enable the OpenACC block and block thread back-end" OFF)
option(alpaka_ACC_CPU_DISABLE_ATOMIC_REF "Disable boost::atomic_ref for CPU back-ends" OFF)
option(alpaka_ACC_SYCL_ENABLE "Enable the SYCL back-end" OFF)

# Unified compiler options
alpaka_compiler_option(FAST_MATH "Enable fast-math" DEFAULT)
alpaka_compiler_option(FTZ "Set flush to zero" DEFAULT)

if((alpaka_ACC_GPU_CUDA_ONLY_MODE OR alpaka_ACC_GPU_HIP_ONLY_MODE)
   AND
    (alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
    alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OR
    alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
    alpaka_ACC_ANY_BT_OMP5_ENABLE OR
    alpaka_ACC_SYCL_ENABLE))
    if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
        message(FATAL_ERROR "If alpaka_ACC_GPU_CUDA_ONLY_MODE is enabled, only back-ends using CUDA can be enabled! This allows to mix alpaka code with native CUDA code. However, this prevents any non-CUDA back-ends from being enabled.")
    endif()
    if(alpaka_ACC_GPU_HIP_ONLY_MODE)
        message(FATAL_ERROR "If alpaka_ACC_GPU_HIP_ONLY_MODE is enabled, only back-ends using HIP can be enabled!")
    endif()
    set(_alpaka_FOUND FALSE)
elseif(alpaka_ACC_ANY_BT_OACC_ENABLE)
    if((alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
       alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
       alpaka_ACC_ANY_BT_OMP5_ENABLE))
       message(WARNING "If alpaka_ACC_ANY_BT_OACC_ENABLE is enabled no OpenMP backend can be enabled.")
    endif()
endif()

# avoids CUDA+HIP conflict
if(alpaka_ACC_GPU_HIP_ENABLE AND alpaka_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "CUDA and HIP can not be enabled both at the same time.")
endif()

# HIP is only supported on Linux
if(alpaka_ACC_GPU_HIP_ENABLE AND (MSVC OR WIN32))
    message(FATAL_ERROR "Optional alpaka dependency HIP can not be built on Windows!")
endif()

# Drop-down combo box in cmake-gui.
set(alpaka_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE alpaka_DEBUG PROPERTY STRINGS "0;1;2")

set(alpaka_CXX_STANDARD_DEFAULT "17")
# Check whether alpaka_CXX_STANDARD has already been defined as a non-cached variable.
if(DEFINED alpaka_CXX_STANDARD)
    set(alpaka_CXX_STANDARD_DEFAULT ${alpaka_CXX_STANDARD})
endif()

set(alpaka_CXX_STANDARD ${alpaka_CXX_STANDARD_DEFAULT} CACHE STRING "C++ standard version")
set_property(CACHE alpaka_CXX_STANDARD PROPERTY STRINGS "17;20")

if(NOT TARGET alpaka)
    add_library(alpaka INTERFACE)

    target_compile_features(alpaka INTERFACE cxx_std_${alpaka_CXX_STANDARD})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        # Workaround for STL atomic issue: https://forums.developer.nvidia.com/t/support-for-atomic-in-libstdc-missing/135403/2
        # still appears in NVHPC 20.7
        target_compile_definitions(alpaka INTERFACE "__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1")
        # reports many unused variables declared by catch test macros
        target_compile_options(alpaka INTERFACE "--diag_suppress 177")
        # prevent NVHPC from warning about unreachable code sections. TODO: Remove this line once the
        # alpaka_UNUSED macro has been removed (dropping support for CUDA < 11.5).
        target_compile_options(alpaka INTERFACE "--diag_suppress 111")
    endif()

    add_library(alpaka::alpaka ALIAS alpaka)
endif()

set(alpaka_OFFLOAD_MAX_BLOCK_SIZE "256" CACHE STRING "Maximum number threads per block to be suggested by any target offloading backends ANY_BT_OMP5 and ANY_BT_OACC.")
option(alpaka_DEBUG_OFFLOAD_ASSUME_HOST "Allow host-only contructs like assert in offload code in debug mode." ON)
set(alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB "47" CACHE STRING "Kibibytes (1024B) of memory to allocate for block shared memory for backends requiring static allocation (includes CPU_B_OMP2_T_SEQ, CPU_B_TBB_T_SEQ, CPU_B_SEQ_T_SEQ, SYCL)")

set(alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM "OFF" CACHE STRING "Whether to use OMP5 built-in directives for block-shared memory and how to treat dynamic shared memory.")
set_property(CACHE alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM PROPERTY STRINGS "OFF;DYN_FIXED;DYN_ALLOC")

#-------------------------------------------------------------------------------
# Debug output of common variables.
if(${alpaka_DEBUG} GREATER 1)
    cmake_print_variables(_alpaka_ROOT_DIR)
    cmake_print_variables(_alpaka_COMMON_FILE)
    cmake_print_variables(_alpaka_ADD_EXECUTABLE_FILE)
    cmake_print_variables(_alpaka_ADD_LIBRARY_FILE)
    cmake_print_variables(CMAKE_BUILD_TYPE)
endif()

#-------------------------------------------------------------------------------
# Check supported compilers.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
    message(FATAL_ERROR "Clang versions < 4.0 are not supported!")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(WARNING "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)!")
endif()

if(alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE AND (alpaka_ACC_GPU_CUDA_ENABLE OR alpaka_ACC_GPU_HIP_ENABLE))
    message(FATAL_ERROR "Fibers and CUDA or HIP back-end can not be enabled both at the same time.")
endif()

if (alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    message(FATAL_ERROR "Clang versions < 6.0 do not support Boost.Fiber!")
endif()

#-------------------------------------------------------------------------------
# Compiler settings.

if(MSVC)
    # CUDA\v9.2\include\crt/host_runtime.h(265): warning C4505: '__cudaUnregisterBinaryUtil': unreferenced local function has been removed
    if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
        target_compile_options(alpaka INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4505>)
    endif()
else()
    # For std::future we need to pass the correct pthread flag for the compiler and the linker:
    # https://github.com/alpaka-group/cupla/pull/128#issuecomment-545078917
    
    # Allow users to override the "-pthread" preference.
    if(NOT THREADS_PREFER_PTHREAD_FLAG)
        set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    endif()
    
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
set(_alpaka_BOOST_MIN_VER "1.74.0")

if(${alpaka_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
endif()

find_package(Boost ${_alpaka_BOOST_MIN_VER} REQUIRED
             OPTIONAL_COMPONENTS atomic fiber)

target_link_libraries(alpaka INTERFACE Boost::headers)

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
   alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OR
   alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)

    if(NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF)
        # Check for C++20 std::atomic_ref first
        if(${alpaka_CXX_STANDARD} VERSION_GREATER_EQUAL "20")
            try_compile(alpaka_HAS_STD_ATOMIC_REF # Result stored here
                        "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
                        SOURCES "${_alpaka_FEATURE_TESTS_DIR}/StdAtomicRef.cpp" # Source file
                        CXX_STANDARD 20
                        CXX_STANDARD_REQUIRED TRUE
                        CXX_EXTENSIONS FALSE)
            if(alpaka_HAS_STD_ATOMIC_REF AND (NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF))
                message(STATUS "std::atomic_ref<T> found")
                target_compile_definitions(alpaka INTERFACE ALPAKA_HAS_STD_ATOMIC_REF)
            else()
                message(STATUS "std::atomic_ref<T> NOT found")
            endif()
        endif()

        if(Boost_ATOMIC_FOUND AND (NOT alpaka_HAS_STD_ATOMIC_REF))
            message(STATUS "boost::atomic_ref<T> found")
            target_link_libraries(alpaka INTERFACE Boost::atomic)
        endif()
    endif()

    if(alpaka_ACC_CPU_DISABLE_ATOMIC_REF OR ((NOT alpaka_HAS_STD_ATOMIC_REF) AND (NOT Boost_ATOMIC_FOUND)))
        message(STATUS "atomic_ref<T> was not found or manually disabled. Falling back to lock-based CPU atomics.")
        target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_ATOMIC_ATOMICREF)
    endif()
endif()

if(alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    if(NOT Boost_FIBER_FOUND)
        message(FATAL_ERROR "Optional alpaka dependency Boost.Fiber could not be found!")
    endif()
endif()

if(${alpaka_DEBUG} GREATER 1)
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
# If available, use C++20 math constants. Otherwise, fall back to M_PI etc.
if(${alpaka_CXX_STANDARD} VERSION_LESS "20")
    set(alpaka_HAS_STD_MATH_CONSTANTS FALSE)
else()
    try_compile(alpaka_HAS_STD_MATH_CONSTANTS # Result stored here
                "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
                SOURCES "${_alpaka_FEATURE_TESTS_DIR}/MathConstants.cpp" # Source file
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED TRUE
                CXX_EXTENSIONS FALSE)
endif()

if(NOT alpaka_HAS_STD_MATH_CONSTANTS)
    message(STATUS "C++20 math constants not found. Falling back to non-standard constants.")
    # Enable non-standard constants for MSVC.
    target_compile_definitions(alpaka INTERFACE "$<$<OR:$<CXX_COMPILER_ID:MSVC>,$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>>:_USE_MATH_DEFINES>")
endif()

#-------------------------------------------------------------------------------
# Find TBB.
if(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    find_package(TBB 2021.4.0.0 REQUIRED)
    target_link_libraries(alpaka INTERFACE TBB::tbb)
endif()

#-------------------------------------------------------------------------------
# Find OpenMP.
if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR alpaka_ACC_ANY_BT_OMP5_ENABLE)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        if(alpaka_ACC_ANY_BT_OMP5_ENABLED)
            if(OpenMP_CXX_VERSION VERSION_LESS 5.0)
                message(FATAL_ERROR "alpaka_ACC_ANY_BT_OMP5_ENABLE requires compiler support for OpenMP 5.0.")

                if((${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 12.0.5))
                    message(FATAL_ERROR "The OpenMP 5.0 back-end requires Xcode 12.5 or later")
                elseif((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 11.0))
                    message(FATAL_ERROR "The OpenMP 5.0 back-end requires clang 11.0 or later")
                endif()
            endif()
        endif()

        target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)

        # Clang versions support OpenMP 5.0 only when given the corresponding flag
        if(alpaka_ACC_ANY_BT_OMP5_ENABLE)
            target_link_options(alpaka INTERFACE $<$<CXX_COMPILER_ID:AppleClang,Clang>:-fopenmp-version=50>)
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency OpenMP could not be found!")
    endif()
endif()

if(alpaka_ACC_ANY_BT_OACC_ENABLE)
   find_package(OpenACC)
   if(OpenACC_CXX_FOUND)
      target_compile_options(alpaka INTERFACE ${OpenACC_CXX_OPTIONS})
      target_link_options(alpaka INTERFACE ${OpenACC_CXX_OPTIONS})
   endif()
endif()

#-------------------------------------------------------------------------------
# Find CUDA.
if(alpaka_ACC_GPU_CUDA_ENABLE)
    # Save the user-defined host compiler (if any)
    set(_alpaka_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            # Use user selected CMake CXX compiler or CMAKE_CUDA_HOST_COMPILER as cuda host compiler to avoid fallback to the default system CXX host compiler.
            # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definition passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
            # ignored by CMake (looks like a CMake bug).
            if(_alpaka_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER ${_alpaka_CUDA_HOST_COMPILER})
            elseif("$ENV{CUDAHOSTCXX}" STREQUAL "")
                set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
            endif()
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        if(alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
            message(FATAL_ERROR "CUDA cannot be used together with Boost.Fiber!")
        endif()

        target_compile_features(alpaka INTERFACE cuda_std_${alpaka_CXX_STANDARD})

        alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)
        alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
        alpaka_compiler_option(CUDA_EXPT_EXTENDED_LAMBDA "Enable experimental, extended host-device lambdas in CUDA with nvcc" ON)

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            message(STATUS "clang is used as CUDA compiler")

            if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
                message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 2!")
            endif()
            if(alpaka_ACC_ANY_BT_OMP5_ENABLE)
                message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 5!")
            endif()

            # libstdc++ since version 7 when GNU extensions are enabled (e.g. -std=gnu++11)
            # uses `__CUDACC__` to avoid defining overloads using non-standard `__float128`.
            # This is fixed in clang-11: https://github.com/llvm/llvm-project/commit/8e20516540444618ad32dd11e835c05804053697
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
                target_compile_definitions(alpaka INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>)
            endif()

            if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 11.0)
                target_compile_options(alpaka INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-cuda-version>)
            endif()

            # This flag silences the warning produced by the Dummy.cpp files:
            # clang: warning: argument unused during compilation: '--cuda-gpu-arch=sm_XX'
            # This seems to be a false positive as all flags are 'unused' for an empty file.
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Qunused-arguments>)

            # Silences warnings that are produced by boost because clang is not correctly identified.
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-local-typedef>)

            if(alpaka_FAST_MATH STREQUAL ON)
                # -ffp-contract=fast enables the usage of FMA
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math -ffp-contract=fast>)
            endif()

            if(alpaka_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-fcuda-flush-denormals-to-zero>)
            endif()

            if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas=-v>)
            endif()

            if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-save-temps>)
            endif()

        elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            message(STATUS "nvcc is used as CUDA compiler")

            # nvcc sets no linux/__linux macros on OpenPOWER linux
            # nvidia bug id: 2448610
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
                    alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Dlinux>)
                endif()
            endif()

            if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
            endif()
            # This is mandatory because with c++17 many standard library functions we rely on are constexpr (std::min, std::multiplies, ...)
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

            if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-g>)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
            endif()

            if(alpaka_FAST_MATH STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
            endif()

            if(alpaka_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>)
            elseif(alpaka_FTZ STREQUAL OFF)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
            endif()

            if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
            endif()

            if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR alpaka_ACC_ANY_BT_OMP5_ENABLE)
                if(NOT MSVC)
                    alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
                else()
                    alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)
                endif()
            endif()

            # Always add warning/error numbers which can be used for suppressions
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--display_error_number>)

            # avoids warnings on host-device signatured, default constructors/destructors
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored>)

            if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
            endif()

            option(alpaka_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx." OFF)
            if(alpaka_CUDA_SHOW_CODELINES)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx -lineinfo>)

                # This is shaky - We currently don't have a way of checking for the host compiler ID.
                # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
                if(NOT MSVC)
                    alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-rdynamic>)
                endif()
                set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
            endif()
        endif()

        target_link_libraries(alpaka INTERFACE CUDA::cudart)
        target_include_directories(alpaka SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(FATAL_ERROR "Optional alpaka dependency CUDA could not be found!")
    endif()
endif()

#-------------------------------------------------------------------------------
# Find HIP.
if(alpaka_ACC_GPU_HIP_ENABLE)

    # supported HIP version range
    set(_alpaka_HIP_MIN_VER 4.0)
    set(_alpaka_HIP_MAX_VER 5.0)
    find_package(hip "${_alpaka_HIP_MAX_VER}")
    if(NOT TARGET hip)
        message(STATUS "Could not find HIP v${_alpaka_HIP_MAX_VER}. Now searching for HIP v${_alpaka_HIP_MIN_VER}")
        find_package(hip "${_alpaka_HIP_MIN_VER}")
    endif()

    if(NOT TARGET hip)
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    else()
        target_link_libraries(alpaka INTERFACE hip::host hip::device)

        alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "-ffast-math")
        endif()

        # hiprand requires ROCm implementation of random numbers by rocrand
        # hip::hiprand is currently not expressing this dependency
        find_package(rocrand REQUIRED CONFIG
                HINTS "${HIP_ROOT_DIR}/rocrand"
                HINTS "/opt/rocm/rocrand")
        if(rocrand_FOUND)
            target_link_libraries(alpaka INTERFACE roc::rocrand)
        else()
            MESSAGE(FATAL_ERROR "Could not find rocRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/rocrand).")
        endif()

        # # HIP random numbers
        find_package(hiprand REQUIRED CONFIG
                HINTS "${HIP_ROOT_DIR}/hiprand"
                HINTS "/opt/rocm/hiprand")
        if(hiprand_FOUND)
            target_link_libraries(alpaka INTERFACE hip::hiprand)
        else()
            MESSAGE(FATAL_ERROR "Could not find hipRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/hiprand).")
        endif()

        alpaka_set_compiler_options(HOST_DEVICE target alpaka -std=c++${alpaka_CXX_STANDARD})

        if(alpaka_HIP_KEEP_FILES STREQUAL ON)
            alpaka_set_compiler_options(HOST_DEVICE target alpaka -save-temps)
        endif()
    endif()

endif() # HIP

#-------------------------------------------------------------------------------
# SYCL settings
if(alpaka_ACC_SYCL_ENABLE)
    # Possible SYCL platforms
    cmake_dependent_option(alpaka_SYCL_PLATFORM_ONEAPI "Enable Intel oneAPI platform for the SYCL back-end" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
    cmake_dependent_option(alpaka_SYCL_PLATFORM_XILINX "Enable Xilinx platform for the SYCL back-end" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
    # Possible oneAPI targets
    cmake_dependent_option(alpaka_SYCL_ONEAPI_CPU "Enable oneAPI CPU targets for the SYCL back-end" OFF "alpaka_SYCL_PLATFORM_ONEAPI" OFF)
    cmake_dependent_option(alpaka_SYCL_ONEAPI_FPGA "Enable oneAPI FPGA targets for the SYCL back-end" OFF "alpaka_SYCL_PLATFORM_ONEAPI" OFF)
    cmake_dependent_option(alpaka_SYCL_ONEAPI_GPU "Enable oneAPI GPU targets for the SYCL back-end" OFF "alpaka_SYCL_PLATFORM_ONEAPI" OFF)
    # Intel FPGA emulation / simulation
    if(alpaka_SYCL_ONEAPI_FPGA)
        set(alpaka_SYCL_ONEAPI_FPGA_MODE "emulation" CACHE STRING "Synthesis type for oneAPI FPGA targets")
        set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_MODE PROPERTY STRINGS "emulation;simulation;hardware")
    endif()
    # Xilinx FPGA emulation / synthesis
    if(alpaka_SYCL_PLATFORM_XILINX)
        set(alpaka_SYCL_XILINX_FPGA_MODE "simulation" CACHE STRING "Synthesis type for Xilinx FPGA targets")
        set_property(CACHE alpaka_SYCL_XILINX_FPGA_MODE PROPERTY STRINGS "simulation;hardware")
    endif()

    # Enable device-side printing to stdout
    cmake_dependent_option(alpaka_SYCL_ENABLE_IOSTREAM "Enable device-side printing to stdout" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
    if(BUILD_TESTING)
        set(alpaka_SYCL_ENABLE_IOSTREAM ON CACHE BOOL "Enable device-side printing to stdout" FORCE)
    endif()

    if(NOT (alpaka_SYCL_PLATFORM_ONEAPI OR alpaka_SYCL_PLATFORM_XILINX))
        message(FATAL_ERROR "You must specify at least one SYCL platform!")
    endif()

    alpaka_set_compiler_options(HOST_DEVICE target alpaka "-fsycl")
    target_link_options(alpaka INTERFACE "-fsycl")
    alpaka_set_compiler_options(HOST_DEVICE target alpaka "-sycl-std=2020")

    #-----------------------------------------------------------------------------------------------------------------
    # Determine SYCL targets
    set(alpaka_SYCL_ONEAPI_CPU_TARGET "spir64_x86_64")
    set(alpaka_SYCL_ONEAPI_FPGA_TARGET "spir64_fpga")
    set(alpaka_SYCL_ONEAPI_GPU_TARGET "spir64_gen")
    set(alpaka_SYCL_XILINX_FPGA_HARDWARE_EMULATION_TARGET "fpga64_hls_hw_emu")
    set(alpaka_SYCL_XILINX_FPGA_HARDWARE_TARGET "fpga64_hls_hw")

    if(alpaka_SYCL_PLATFORM_ONEAPI)
        if(alpaka_SYCL_ONEAPI_CPU)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_CPU_TARGET})
        endif()

        if(alpaka_SYCL_ONEAPI_FPGA)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_FPGA_TARGET})
        endif()

        if(alpaka_SYCL_ONEAPI_GPU)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_GPU_TARGET})
        endif()

        if(NOT alpaka_SYCL_TARGETS)
            message(FATAL_ERROR "You must specify at least one oneAPI hardware target!")
        endif()
    endif()

    if(alpaka_SYCL_PLATFORM_XILINX)
        if(alpaka_SYCL_XILINX_FPGA_MODE STREQUAL "simulation")
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_XILINX_FPGA_HARDWARE_EMULATION_TARGET})
        elseif(alpaka_SYCL_XILINX_FPGA_MODE STREQUAL "hardware")
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_XILINX_FPGA_HARDWARE_TARGET})
        else()
            message(FATAL_ERROR "You must specify at least one Xilinx FPGA target!")
        endif()
    endif()

    list(JOIN alpaka_SYCL_TARGETS "," alpaka_SYCL_TARGETS_CONCAT)
    alpaka_set_compiler_options(HOST_DEVICE target alpaka "-fsycl-targets=${alpaka_SYCL_TARGETS_CONCAT}")
    target_link_options(alpaka INTERFACE "-fsycl-targets=${alpaka_SYCL_TARGETS_CONCAT}")
    
    #-----------------------------------------------------------------------------------------------------------------
    # Determine actual hardware to compile for 
    if(alpaka_SYCL_ONEAPI_CPU)
        set(alpaka_SYCL_ONEAPI_CPU_ISA "avx2" CACHE STRING "Intel ISA to compile for")
        set_property(CACHE alpaka_SYCL_ONEAPI_CPU_ISA PROPERTY STRINGS "sse4.2;avx;avx2;avx512")

        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_CPU")
        target_link_options(alpaka INTERFACE "SHELL:-Xsycl-target-backend=${alpaka_SYCL_ONEAPI_CPU_TARGET} \"-march=${alpaka_SYCL_ONEAPI_CPU_ISA}\"")
    endif()

    if(alpaka_SYCL_ONEAPI_FPGA)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_FPGA")
        # try to come as close to -fintelfpga as possible with the following two flags
        alpaka_set_compiler_options(DEVICE target alpaka "-fintelfpga")

        if(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "emulation")
            target_compile_definitions(alpaka INTERFACE "ALPAKA_FPGA_EMULATION")
            alpaka_set_compiler_options(DEVICE target alpaka "-Xsemulator")
            target_link_options(alpaka INTERFACE "-Xsemulator")
        elseif(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "simulation")
            alpaka_set_compiler_options(DEVICE target alpaka "-Xssimulation")
            target_link_options(alpaka INTERFACE "-Xssimulation")
        elseif(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "hardware")
            alpaka_set_compiler_options(DEVICE target alpaka "-Xshardware")
            target_link_options(alpaka INTERFACE "-Xshardware")
        endif()

        if(NOT alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "emulation")
            set(alpaka_SYCL_ONEAPI_FPGA_BOARD "pac_a10" CACHE STRING "Intel FPGA board to compile for")
            set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_BOARD PROPERTY STRINGS "pac_a10;pac_s10;pac_s10_usm")

            set(alpaka_SYCL_ONEAPI_FPGA_BSP "intel_a10gx_pac" CACHE STRING "Path to or name of the Intel FPGA board support package")
            set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_BSP PROPERTY STRINGS "intel_a10gx_pac;intel_s10sx_pac")
            target_link_options(alpaka INTERFACE "-Xsycl-target-backend=${alpaka_SYCL_ONEAPI_FPGA_TARGET} \"-board=${alpaka_SYCL_ONEAPI_FPGA_BSP}:${alpaka_SYCL_ONEAPI_FPGA_BOARD}\"")
        endif()

    endif()

    if(alpaka_SYCL_ONEAPI_GPU)
        # Create a drop-down list (in cmake-gui) of valid Intel GPU targets. On the command line the user can specifiy
        # additional targets, such as ranges: "Gen8-Gen12LP" or lists: "icclp;skl".
        set(alpaka_SYCL_ONEAPI_GPU_DEVICES "bdw" CACHE STRING "Intel GPU devices / generations to compile for")
        set_property(CACHE alpaka_SYCL_ONEAPI_GPU_DEVICES
                     PROPERTY STRINGS "bdw;skl;kbl;cfl;bxt;glk;icllp;lkf;ehl;tgllp;rkl;adls;adlp;dg1;xe_hp_sdv;Gen8;Gen9;Gen11;Gen12LP;XE_HP_CORE")
        # If the user has given us a list turn all ';' into ',' to pacify the Intel OpenCL compiler.
        string(REPLACE alpaka_ONEAPI_GPU_DEVICES REPLACE ";" ",")
        
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_GPU")
        target_link_options(alpaka INTERFACE "SHELL:-Xsycl-target-backend=${alpaka_SYCL_ONEAPI_GPU_TARGET} \"-device ${alpaka_SYCL_ONEAPI_GPU_DEVICES}\"")
    endif()

    #-----------------------------------------------------------------------------------------------------------------
    # Generic SYCL options
    if(alpaka_SYCL_ENABLE_IOSTREAM)
        set(alpaka_SYCL_IOSTREAM_KIB "64" CACHE STRING "Kibibytes (1024B) of memory to allocate per block for SYCL's on-device iostream")

        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_IOSTREAM_ENABLED")
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_IOSTREAM_KIB=${alpaka_SYCL_IOSTREAM_KIB}")
    endif()
    alpaka_set_compiler_options(DEVICE target alpaka "-fsycl-unnamed-lambda") # Compiler default but made explicit here
endif()

#-------------------------------------------------------------------------------
# alpaka.
if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ONLY_MODE")
    message(STATUS alpaka_ACC_GPU_CUDA_ONLY_MODE)
endif()

if(alpaka_ACC_GPU_HIP_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ONLY_MODE")
    message(STATUS alpaka_ACC_GPU_HIP_ONLY_MODE)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
endif()
if(alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED")
    target_link_libraries(alpaka INTERFACE Boost::fiber)
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
endif()
if(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLED)
endif()
if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()
if(alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
endif()
if(alpaka_ACC_ANY_BT_OMP5_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_ANY_BT_OMP5_ENABLED")
    message(STATUS alpaka_ACC_ANY_BT_OMP5_ENABLED)
endif()
if(alpaka_ACC_ANY_BT_OACC_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_ANY_BT_OACC_ENABLED")
    message(STATUS alpaka_ACC_ANY_BT_OACC_ENABLE)
endif()
if(alpaka_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ENABLED")
    message(STATUS alpaka_ACC_GPU_CUDA_ENABLED)
endif()
if(alpaka_ACC_GPU_HIP_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ENABLED")
    message(STATUS alpaka_ACC_GPU_HIP_ENABLED)
endif()

if(alpaka_ACC_SYCL_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_SYCL_ENABLED")
    if(alpaka_SYCL_PLATFORM_ONEAPI)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_BACKEND_ONEAPI")
        if(alpaka_SYCL_ONEAPI_CPU)
            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_CPU")
        endif()
        if(alpaka_SYCL_ONEAPI_FPGA)
            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_FPGA")
        endif()
        if(alpaka_SYCL_ONEAPI_GPU)
            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_GPU")
        endif()
    endif()

    if(alpaka_SYCL_PLATFORM_XILINX)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_BACKEND_XILINX")
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_FPGA")
    endif()

    message(STATUS alpaka_ACC_SYCL_ENABLED)
endif()

target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG=${alpaka_DEBUG}")
if(alpaka_DEBUG_OFFLOAD_ASSUME_HOST)
   target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST")
endif()
target_compile_definitions(alpaka INTERFACE "ALPAKA_OFFLOAD_MAX_BLOCK_SIZE=${alpaka_OFFLOAD_MAX_BLOCK_SIZE}")
target_compile_definitions(alpaka INTERFACE "ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=${alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB}")
if(alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM STREQUAL "DYN_FIXED")
    target_compile_definitions(alpaka INTERFACE ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_FIXED)
elseif(alpaka_OFFLOAD_USE_BUILTIN_SHARED_MEM STREQUAL "DYN_ALLOC")
    target_compile_definitions(alpaka INTERFACE ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_ALLOC)
endif()

if(alpaka_CI)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_CI")
endif()

#-------------------------------------------------------------------------------
# Target.
if(TARGET alpaka)

    if(${alpaka_DEBUG} GREATER 1)
        # Compile options.
        get_property(_alpaka_COMPILE_OPTIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_OPTIONS)
        cmake_print_variables(_alpaka_COMPILE_OPTIONS_PUBLIC)

        # Compile definitions
        get_property(_alpaka_COMPILE_DEFINITIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_DEFINITIONS)
        cmake_print_variables(_alpaka_COMPILE_DEFINITIONS_PUBLIC)

        # Include directories.
        get_property(_alpaka_INCLUDE_DIRECTORIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        cmake_print_variables(_alpaka_INCLUDE_DIRECTORIES_PUBLIC)
    endif()

    # the alpaka library itself
    # SYSTEM voids showing warnings produced by alpaka when used in user applications.
    if(BUILD_TESTING)
        target_include_directories(alpaka INTERFACE ${_alpaka_INCLUDE_DIRECTORY})
    else()
        target_include_directories(alpaka SYSTEM INTERFACE ${_alpaka_INCLUDE_DIRECTORY})
    endif()

    if(${alpaka_DEBUG} GREATER 1)
        # Link libraries.
        # There are no PUBLIC_LINK_FLAGS in CMAKE:
        # http://stackoverflow.com/questions/26850889/cmake-keeping-link-flags-of-internal-libs
        get_property(_alpaka_LINK_LIBRARIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_LIBRARIES)
        cmake_print_variables(_alpaka_LINK_LIBRARIES_PUBLIC)

        get_property(_alpaka_LINK_FLAGS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_OPTIONS)
        cmake_print_variables(_alpaka_LINK_FLAGS_PUBLIC)
    endif()
endif()

# Compile options summary
if(alpaka_COMPILER_OPTIONS_DEVICE OR alpaka_COMPILER_OPTIONS_DEVICE)
    message("")
    message("List of compiler flags added by alpaka")
    if(alpaka_COMPILER_OPTIONS_HOST)
        message("host compiler:")
        message("    ${alpaka_COMPILER_OPTIONS_HOST}")
    endif()
    if(alpaka_COMPILER_OPTIONS_DEVICE)
        message("device compiler:")
        message("    ${alpaka_COMPILER_OPTIONS_DEVICE}")
    endif()
    message("")
endif()
