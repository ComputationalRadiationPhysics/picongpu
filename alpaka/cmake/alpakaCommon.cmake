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

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED ALPAKA_${name})
        set(ALPAKA_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE ALPAKA_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# ALPAKA_COMPILER_OPTIONS_HOST, ALPAKA_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(ALPAKA_COMPILER_OPTIONS_HOST ${ALPAKA_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(ALPAKA_COMPILER_OPTIONS_DEVICE ${ALPAKA_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(ALPAKA_COMPILER_OPTIONS_HOST ${ALPAKA_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(ALPAKA_COMPILER_OPTIONS_DEVICE ${ALPAKA_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
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
option(ALPAKA_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)
option(ALPAKA_ACC_GPU_HIP_ONLY_MODE "Only back-ends using HIP can be enabled in this mode." OFF) # HIP only runs without other back-ends

option(ALPAKA_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" OFF)
option(ALPAKA_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode (This allows to mix alpaka code with native CUDA code)." OFF)

if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE AND NOT ALPAKA_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "If ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled, ALPAKA_ACC_GPU_CUDA_ENABLE has to be enabled as well.")
endif()
if(ALPAKA_ACC_GPU_HIP_ONLY_MODE AND NOT ALPAKA_ACC_GPU_HIP_ENABLE)
    message(FATAL_ERROR "If ALPAKA_ACC_GPU_HIP_ONLY_MODE is enabled, ALPAKA_ACC_GPU_HIP_ENABLE has to be enabled as well.")
endif()

option(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" OFF)
option(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" OFF)
option(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread back-end" OFF)
option(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" OFF)
option(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" OFF)
option(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" OFF)
option(ALPAKA_ACC_ANY_BT_OMP5_ENABLE "Enable the OpenMP 5.0 CPU block and block thread back-end" OFF)
option(ALPAKA_ACC_ANY_BT_OACC_ENABLE "Enable the OpenACC block and block thread back-end" OFF)

# Unified compiler options
alpaka_compiler_option(FAST_MATH "Enable fast-math" DEFAULT)
alpaka_compiler_option(FTZ "Set flush to zero" DEFAULT)

if((ALPAKA_ACC_GPU_CUDA_ONLY_MODE OR ALPAKA_ACC_GPU_HIP_ONLY_MODE)
   AND
    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OR
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
    ALPAKA_ACC_ANY_BT_OMP5_ENABLE))
    if(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
        message(FATAL_ERROR "If ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled, only back-ends using CUDA can be enabled! This allows to mix alpaka code with native CUDA code. However, this prevents any non-CUDA back-ends from being enabled.")
    endif()
    if(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
        message(FATAL_ERROR "If ALPAKA_ACC_GPU_HIP_ONLY_MODE is enabled, only back-ends using HIP can be enabled!")
    endif()
    set(_ALPAKA_FOUND FALSE)
elseif(ALPAKA_ACC_ANY_BT_OACC_ENABLE)
    if((ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
       ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
       ALPAKA_ACC_ANY_BT_OMP5_ENABLE))
       message(WARNING "If ALPAKA_ACC_ANY_BT_OACC_ENABLE is enabled no OpenMP backend can be enabled.")
    endif()
endif()

# avoids CUDA+HIP conflict
if(ALPAKA_ACC_GPU_HIP_ENABLE AND ALPAKA_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "CUDA and HIP can not be enabled both at the same time.")
endif()

# HIP is only supported on Linux
if(ALPAKA_ACC_GPU_HIP_ENABLE AND (MSVC OR WIN32))
    message(FATAL_ERROR "Optional alpaka dependency HIP can not be built on Windows!")
endif()

# Drop-down combo box in cmake-gui.
set(ALPAKA_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE ALPAKA_DEBUG PROPERTY STRINGS "0;1;2")

set(ALPAKA_CXX_STANDARD "14" CACHE STRING "C++ standard version")
set_property(CACHE ALPAKA_CXX_STANDARD PROPERTY STRINGS "14;17;20")

if(NOT TARGET alpaka)
    add_library(alpaka INTERFACE)

    target_compile_features(alpaka INTERFACE cxx_std_${ALPAKA_CXX_STANDARD})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        # Workaround for STL atomic issue: https://forums.developer.nvidia.com/t/support-for-atomic-in-libstdc-missing/135403/2
        # still appears in NVHPC 20.7
        target_compile_definitions(alpaka INTERFACE "__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1")
    endif()

    add_library(alpaka::alpaka ALIAS alpaka)
endif()

set(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE "256" CACHE STRING "Maximum number threads per block to be suggested by any target offloading backends ANY_BT_OMP5 and ANY_BT_OACC.")
option(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST "Allow host-only contructs like assert in offload code in debug mode." ON)
set(ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB "47" CACHE STRING "Kibibytes (1024B) of memory to allocate for block shared memory for backends requiring static allocation (includes CPU_B_OMP2_T_SEQ, CPU_B_TBB_T_SEQ, CPU_B_SEQ_T_SEQ)")

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
set(_ALPAKA_BOOST_MIN_VER "1.65.1")

if(${ALPAKA_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
endif()

find_package(Boost ${_ALPAKA_BOOST_MIN_VER} REQUIRED
             OPTIONAL_COMPONENTS fiber)

target_link_libraries(alpaka INTERFACE Boost::headers)

if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    if(NOT Boost_FIBER_FOUND)
        message(FATAL_ERROR "Optional alpaka dependency Boost.Fiber could not be found!")
    endif()
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
    # Prefer TBB's own TBBConfig.cmake (available in more recent TBB versions)
    find_package(TBB QUIET CONFIG)
    
    if(NOT TBB_FOUND)
        message(STATUS "TBB not found in config mode. Retrying in module mode.")
        find_package(TBB REQUIRED MODULE)
    endif()

    target_link_libraries(alpaka INTERFACE TBB::tbb)
endif()

#-------------------------------------------------------------------------------
# Find OpenMP.
if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR ALPAKA_ACC_ANY_BT_OMP5_ENABLE)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        if(ALPAKA_ACC_ANY_BT_OMP5_ENABLED)
            if(OpenMP_CXX_VERSION VERSION_LESS 5.0)
                message(FATAL_ERROR "ALPAKA_ACC_ANY_BT_OMP5_ENABLE requires compiler support for OpenMP 5.0.")

                if((${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 12.0.5))
                    message(FATAL_ERROR "The OpenMP 5.0 back-end requires Xcode 12.5 or later")
                elseif((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 11.0))
                    message(FATAL_ERROR "The OpenMP 5.0 back-end requires clang 11.0 or later")
                endif()
            endif()
        endif()

        target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)

        # Clang versions support OpenMP 5.0 only when given the corresponding flag
        if(ALPAKA_ACC_ANY_BT_OMP5_ENABLE)
            target_link_options(alpaka INTERFACE $<$<CXX_COMPILER_ID:AppleClang,Clang>:-fopenmp-version=50>)
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency OpenMP could not be found!")
    endif()
endif()

if(ALPAKA_ACC_ANY_BT_OACC_ENABLE)
   find_package(OpenACC)
   if(OpenACC_CXX_FOUND)
      target_compile_options(alpaka INTERFACE ${OpenACC_CXX_OPTIONS})
      target_link_options(alpaka INTERFACE ${OpenACC_CXX_OPTIONS})
   endif()
endif()

#-------------------------------------------------------------------------------
# Find CUDA.
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    # Save the user-defined host compiler (if any)
    set(_ALPAKA_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            # Use user selected CMake CXX compiler or CMAKE_CUDA_HOST_COMPILER as cuda host compiler to avoid fallback to the default system CXX host compiler.
            # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definition passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
            # ignored by CMake (looks like a CMake bug).
            if(_ALPAKA_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER ${_ALPAKA_CUDA_HOST_COMPILER})
            elseif("$ENV{CUDAHOSTCXX}" STREQUAL "")
                set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
            endif()
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        if(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
            message(FATAL_ERROR "CUDA cannot be used together with Boost.Fiber!")
        endif()

        target_compile_features(alpaka INTERFACE cuda_std_${ALPAKA_CXX_STANDARD})

        alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)
        alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
        alpaka_compiler_option(CUDA_EXPT_EXTENDED_LAMBDA "Enable experimental, extended host-device lambdas in CUDA with nvcc" ON)

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            message(STATUS "clang is used as CUDA compiler")

            if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
                message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 2!")
            endif()
            if(ALPAKA_ACC_ANY_BT_OMP5_ENABLE)
                message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 5!")
            endif()

            # libstdc++ since version 7 when GNU extensions are enabled (e.g. -std=gnu++11)
            # uses `__CUDACC__` to avoid defining overloads using non-standard `__float128`.
            # This is fixed in clang-11: https://github.com/llvm/llvm-project/commit/8e20516540444618ad32dd11e835c05804053697
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
                target_compile_definitions(alpaka INTERFACE "__CUDACC__")
            endif()

            if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 11.0)
                target_compile_options(alpaka INTERFACE "-Wno-unknown-cuda-version")
            endif()

            # This flag silences the warning produced by the Dummy.cpp files:
            # clang: warning: argument unused during compilation: '--cuda-gpu-arch=sm_XX'
            # This seems to be a false positive as all flags are 'unused' for an empty file.
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Qunused-arguments>)

            # Silences warnings that are produced by boost because clang is not correctly identified.
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-local-typedef>)

            if(ALPAKA_FAST_MATH STREQUAL ON)
                # -ffp-contract=fast enables the usage of FMA
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math -ffp-contract=fast>)
            endif()

            if(ALPAKA_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-fcuda-flush-denormals-to-zero>)
            endif()

            if(ALPAKA_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas=-v>)
            endif()

            if(ALPAKA_CUDA_KEEP_FILES STREQUAL ON)
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

            # NOTE: Since CUDA 10.2 this option is also alternatively called '--extended-lambda'
            if(ALPAKA_CUDA_EXPT_EXTENDED_LAMBDA STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
            endif()
            # This is mandatory because with c++14 many standard library functions we rely on are constexpr (std::min, std::multiplies, ...)
            alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

            if((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-g>)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
            endif()

            if(ALPAKA_FAST_MATH STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
            endif()

            if(ALPAKA_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>)
            elseif(ALPAKA_FTZ STREQUAL OFF)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
            endif()

            if(ALPAKA_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
            endif()

            if(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR ALPAKA_ACC_ANY_BT_OMP5_ENABLE)
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

            # avoids warnings on host-device signature of 'std::__shared_count<>'
            if(CUDAToolkit_VERSION VERSION_EQUAL 10.0)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=2905>)
            elseif(CUDAToolkit_VERSION VERSION_EQUAL 10.1)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=2912>)
            elseif(CUDAToolkit_VERSION VERSION_EQUAL 10.2)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=2976>)
            endif()

            if(ALPAKA_CUDA_KEEP_FILES STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
            endif()

            option(ALPAKA_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck. If ALPAKA_CUDA_KEEP_FILES is enabled source code will be inlined in ptx." OFF)
            if(ALPAKA_CUDA_SHOW_CODELINES)
                alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx -lineinfo>)

                # This is shaky - We currently don't have a way of checking for the host compiler ID.
                # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
                if(NOT MSVC)
                    alpaka_set_compiler_options(DEVICE target alpaka $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-rdynamic>)
                endif()
                set(ALPAKA_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
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
if(ALPAKA_ACC_GPU_HIP_ENABLE)

    # minimal supported HIP version
    set(_ALPAKA_HIP_MIN_VER 4.0)
    find_package(hip "${_ALPAKA_HIP_MIN_VER}")

    if(NOT TARGET hip)
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    else()
        target_link_libraries(alpaka INTERFACE hip::host hip::device)

        alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)

        if(ALPAKA_FAST_MATH STREQUAL ON)
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

        alpaka_set_compiler_options(HOST_DEVICE target alpaka -std=c++${ALPAKA_CXX_STANDARD})

        if(ALPAKA_HIP_KEEP_FILES STREQUAL ON)
            alpaka_set_compiler_options(HOST_DEVICE target alpaka -save-temps)
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
if(ALPAKA_ACC_ANY_BT_OMP5_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_ANY_BT_OMP5_ENABLED")
    message(STATUS ALPAKA_ACC_ANY_BT_OMP5_ENABLED)
endif()
if(ALPAKA_ACC_ANY_BT_OACC_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_ANY_BT_OACC_ENABLED")
    message(STATUS ALPAKA_ACC_ANY_BT_OACC_ENABLE)
endif()
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ENABLED")
    message(STATUS ALPAKA_ACC_GPU_CUDA_ENABLED)
endif()
if(ALPAKA_ACC_GPU_HIP_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ENABLED")
    message(STATUS ALPAKA_ACC_GPU_HIP_ENABLED)
endif()

target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
if(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST)
   target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST")
endif()
target_compile_definitions(alpaka INTERFACE "ALPAKA_OFFLOAD_MAX_BLOCK_SIZE=${ALPAKA_OFFLOAD_MAX_BLOCK_SIZE}")
target_compile_definitions(alpaka INTERFACE "ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=${ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB}")

if(ALPAKA_CI)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_CI")
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
    # SYSTEM voids showing warnings produced by alpaka when used in user applications.
    if(BUILD_TESTING)
        target_include_directories(alpaka INTERFACE ${_ALPAKA_INCLUDE_DIRECTORY})
    else()
        target_include_directories(alpaka SYSTEM INTERFACE ${_ALPAKA_INCLUDE_DIRECTORY})
    endif()

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

# Compile options summary
if(ALPAKA_COMPILER_OPTIONS_DEVICE OR ALPAKA_COMPILER_OPTIONS_DEVICE)
    message("")
    message("List of compiler flags added by alpaka")
    if(ALPAKA_COMPILER_OPTIONS_HOST)
        message("host compiler:")
        message("    ${ALPAKA_COMPILER_OPTIONS_HOST}")
    endif()
    if(ALPAKA_COMPILER_OPTIONS_DEVICE)
        message("device compiler:")
        message("    ${ALPAKA_COMPILER_OPTIONS_DEVICE}")
    endif()
    message("")
endif()
