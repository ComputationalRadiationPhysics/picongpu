#
# Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------
# By marking the boost headers as system headers, warnings produced within them are ignored.
# Marking the boost headers as system headers does not work for nvcc (FindCUDA always uses -I)
TARGET_INCLUDE_DIRECTORIES(
    "alpaka"
    SYSTEM
    INTERFACE ${Boost_INCLUDE_DIRS})

IF(alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"))
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
      LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Wdefault-stream-launch -Werror=default-stream-launch>)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
        # supress error in Catch: 'error #177-D: variable "<unnamed>::autoRegistrar1" was declared but never referenced'
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
    endif()
ENDIF()

#MSVC
IF(MSVC)
    # Force to always compile with W4 and WX
    LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/WX> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    # Improve debugging.
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zo> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zo>)
    ENDIF()
    IF(MSVC_VERSION GREATER 1900)
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/permissive-> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->)
        IF(MSVC_VERSION GREATER 1910)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zc:twoPhase-> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:twoPhase->)
        ENDIF()
    ENDIF()
    IF(MSVC_VERSION GREATER 1800)
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zc:throwingNew> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:throwingNew>)
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zc:strictStrings> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:strictStrings>)
    ENDIF()
ELSE()
  IF(NOT(alpaka_ACC_GPU_CUDA_ENABLE) OR CMAKE_CUDA_COMPILER_ID STREQUAL "Clang"
      OR(alpaka_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wall")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wextra")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-pedantic")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Werror")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdouble-promotion")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wmissing-include-dirs")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconversion")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wunknown-pragmas")
        # Higher levels (max is 5) produce some strange warnings
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wstrict-overflow=2")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wtrampolines")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wfloat-equal")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wundef")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wshadow")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-qual")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-align")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wwrite-strings")
        # Too noisy as it warns for every operation using numeric types smaller then int.
        # Such values are converted to int implicitly before the calculation is done.
        # E.g.: uint16_t = uint16_t * uint16_t will trigger the following warning:
        # conversion to ‘short unsigned int’ from ‘int’ may alter its value
        #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconversion")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsign-conversion")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wvector-operation-performance")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wzero-as-null-pointer-constant")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdate-time")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wuseless-cast")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wlogical-op")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-aggressive-loop-optimizations")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wmissing-declarations")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-multichar")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wopenmp-simd")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wpacked")
        # Too much noise
        #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wpadded")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wredundant-decls")
        # Too much noise
        #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Winline")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdisabled-optimization")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-nonliteral")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-security")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-y2k")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wctor-dtor-privacy")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdelete-non-virtual-dtor")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wliteral-suffix")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnon-virtual-dtor")
        # This warns about members that have not explicitly been listed in the constructor initializer list.
        # This could be useful even for members that have a default constructor.
        # However, it also issues this warning for defaulted constructurs.
        #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Weffc++")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Woverloaded-virtual")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsign-promo")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconditionally-supported")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnoexcept")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wold-style-cast")
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-final-types")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-final-methods")
            # This does not work correctly as it suggests override to methods that are already marked with final.
            # Because final implies override, this is not useful.
            #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-override")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnormalized")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-signedness")
        ENDIF()
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnull-dereference")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wduplicated-cond")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsubobject-linkage")
        ENDIF()
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
            # This warning might be useful but it is triggered by comile-time code where it does not make any sense:
            # E.g. "Vec<DimInt<(TidxDimOut < TidxDimIn) ? TidxDimIn : TidxDimOut>, TElem>" when both values are equal
            #LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wduplicated-branches")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Walloc-zero")
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Walloca")
        ENDIF()
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-align=strict")
        ENDIF()

    # Clang, AppleClang, ICPX
    ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Werror")
        # Weverything really means everything (including Wall, Wextra, pedantic, ...)
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Weverything")
        # We are not C++98 compatible
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-c++98-compat")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-c++98-compat-pedantic")
        # The following warnings are triggered by all instantiations of BOOST_AUTO_TEST_SUITE
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-disabled-macro-expansion")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-global-constructors")
        # This padding warning is generated by the execution tasks depending on the argument types
        # as they are stored as members. Therefore, the padding warning is triggered by the calling code
        # and does not indicate a failure within alpaka.
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-padded")
        # Triggers for all instances of alpaka_DEBUG_MINIMAL_LOG_SCOPE and similar macros followed by semicolon
        IF(CLANG_VERSION_MAJOR GREATER 7)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-extra-semi-stmt")
        ENDIF()
        IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-poison-system-directories")
        ENDIF()
        IF(alpaka_ACC_GPU_HIP_ENABLE)
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-unused-command-line-argument")
        ENDIF()

        SET(IS_ICPX OFF)
        IF(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            SET(IS_ICPX ON)
        ELSEIF (CMAKE_VERSION VERSION_LESS 3.20 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            INCLUDE(CheckCXXSourceCompiles)
            CHECK_CXX_SOURCE_COMPILES("\
                #ifndef __INTEL_LLVM_COMPILER\
                #    error\
                #endif
                int main() {}"
                IS_ICPX)
        ENDIF()
        IF (IS_ICPX)
            # fast math is turned on by default with ICPX, which breaks our unit tests
            LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-ffp-model=precise")
        ENDIF()
    # ICPC
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wall")
    # PGI
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        LIST(APPEND alpaka_DEV_COMPILE_OPTIONS "-Minform=inform")
    ENDIF()
  ENDIF()
ENDIF()
