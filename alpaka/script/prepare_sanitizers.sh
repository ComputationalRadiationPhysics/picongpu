#!/bin/bash

#
# Copyright 2017-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

source ./script/set.sh

#-------------------------------------------------------------------------------
# Exports the CMAKE_CXX_FLAGS and CMAKE_EXE_LINKER_FLAGS to enable the sanitizers listed in ALPAKA_CI_SANITIZERS.
if [ -z "${CMAKE_CXX_FLAGS+x}" ]
then
    export CMAKE_CXX_FLAGS=
fi
if [ -z "${CMAKE_EXE_LINKER_FLAGS+x}" ]
then
    export CMAKE_EXE_LINKER_FLAGS=
fi
if [ -z "${ASAN_OPTIONS+x}" ]
then
    export ASAN_OPTIONS=
fi
if [ -z "${LSAN_OPTIONS+x}" ]
then
    export LSAN_OPTIONS=
fi

#-------------------------------------------------------------------------------
# sanitizers
# General sanitizer settings
if [[ "${ALPAKA_CI_SANITIZERS}" != "" ]]
then
    # - to get nicer stack-traces:
    CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer"
    # - to get perfect stack-traces:
    CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fno-optimize-sibling-calls"

    # g++ needs to use a different linker
    if [[ "${CXX}" == "g++"* ]]
    then
        CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold"
    fi

    # UBSan - http://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
    if [[ "${ALPAKA_CI_SANITIZERS}" == *"UBSan"* ]]
    then
        CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=undefined"

        if [[ "${CXX}" == "clang++"* ]]
        then
            # Previously 'local-bounds' was part of UBsan but has been removed because it is not a pure front-end check
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=local-bounds"
            # 'unsigned-integer-overflow' is not really undefined behaviour but we want to handle it as such for our tests.
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=unsigned-integer-overflow"
        fi
    fi

    # ASan - http://clang.llvm.org/docs/AddressSanitizer.html
    if [[ "${ALPAKA_CI_SANITIZERS}" == *"ASan"* ]]
    then
        if ( [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]] || [[ "${ALPAKA_CI_SANITIZERS}" == *"MSan"* ]] )
        then
            echo ASan is not supported in combination with TSan or MSan
            exit 1
        fi

        if ( [ "${alpaka_ACC_GPU_CUDA_ENABLE}" == "ON" ] && [ "${CMAKE_CUDA_COMPILER}" == "clang++" ] )
        then
            # fatal error: error in backend: Module has a nontrivial global ctor, which NVPTX does not support.
            # clang-3.9: error: clang frontend command failed with exit code 70 (use -v to see invocation)
            echo ASan is not supported in combination with clang used as CUDA compiler
            exit 1
        fi

        CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=address"

        if [[ "${CXX}" != "clang++"* ]]
        then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize-address-use-after-scope"
        fi

        ASAN_OPTIONS="strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1"
    fi

    # TSan - http://clang.llvm.org/docs/ThreadSanitizer.html
    # TSan requires PositionIndependentCode -pie;-fPIE;-fPIC. clang sets this automatically, gcc not.
    # All base libraries (e.g. boost) have to be build with this flag.
    # Furthermore, by installing gcc, libtsan0 is not automatically installed.
    if [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]]
    then
        if ( [[ "${ALPAKA_CI_SANITIZERS}" == *"ASan"* ]] || [[ "${ALPAKA_CI_SANITIZERS}" == *"MSan"* ]] )
        then
            echo TSan is not supported in combination with ASan or MSan
            exit 1
        fi

        CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=thread"
        if [[ "${CXX}" == "g++"* ]]
        then
            CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -pie -fPIE"
            CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -ltsan"
        fi
    fi

    # MSan - http://clang.llvm.org/docs/MemorySanitizer.html
    # NOTE: Currently we can not enable this for CI as this finds some 'use-of-uninitialized-value' inside:
    #   - boost`s smart pointers used by the unit test framework
    #   - alpaka/test/integ/mandelbrot/src/main.cpp:450:9 std::replace
    #   - alpaka/include/alpaka/kernel/TaskKernelCpuThreads.hpp:307:21 used alpaka/include/alpaka/idx/bt/IdxBtRefThreadIdMap.hpp:130:44
    if [[ "${ALPAKA_CI_SANITIZERS}" == *"MSan"* ]]
    then
        if ( [[ "${ALPAKA_CI_SANITIZERS}" == *"ASan"* ]] || [[ "${ALPAKA_CI_SANITIZERS}" == *"TSan"* ]] )
        then
            echo MSan is not supported in combination with ASan or TSan
            exit 1
        fi

        CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins"
    fi
fi
