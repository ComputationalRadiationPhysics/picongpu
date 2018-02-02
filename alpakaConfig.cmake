#
# Copyright 2014-2017 Benjamin Worpitz, Erik Zenker
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version.

CMAKE_MINIMUM_REQUIRED(VERSION 3.7.0)

################################################################################
# alpaka.

# Return values.
UNSET(alpaka_FOUND)
UNSET(alpaka_VERSION)
UNSET(alpaka_COMPILE_OPTIONS)
UNSET(alpaka_COMPILE_DEFINITIONS)
UNSET(alpaka_DEFINITIONS)
UNSET(alpaka_INCLUDE_DIR)
UNSET(alpaka_INCLUDE_DIRS)
UNSET(alpaka_LIBRARY)
UNSET(alpaka_LIBRARIES)

# Internal usage.
UNSET(_ALPAKA_FOUND)
UNSET(_ALPAKA_COMPILE_OPTIONS_PUBLIC)
UNSET(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_ALPAKA_INCLUDE_DIRECTORY)
UNSET(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_ALPAKA_LINK_LIBRARIES_PUBLIC)
UNSET(_ALPAKA_LINK_FLAGS_PUBLIC)
UNSET(_ALPAKA_COMMON_FILE)
UNSET(_ALPAKA_ADD_EXECUTABLE_FILE)
UNSET(_ALPAKA_ADD_LIBRRAY_FILE)
UNSET(_ALPAKA_FILES_HEADER)
UNSET(_ALPAKA_FILES_SOURCE)
UNSET(_ALPAKA_FILES_OTHER)
UNSET(_ALPAKA_VERSION_DEFINE)
UNSET(_ALPAKA_VER_MAJOR)
UNSET(_ALPAKA_VER_MINOR)
UNSET(_ALPAKA_VER_PATCH)

#-------------------------------------------------------------------------------
# Common.

# Directory of this file.
SET(_ALPAKA_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_ALPAKA_ROOT_DIR "${_ALPAKA_ROOT_DIR}" ABSOLUTE)

# Add common functions.
SET(_ALPAKA_COMMON_FILE "${_ALPAKA_ROOT_DIR}/cmake/common.cmake")
INCLUDE("${_ALPAKA_COMMON_FILE}")

# Add ALPAKA_ADD_EXECUTABLE function.
SET(_ALPAKA_ADD_EXECUTABLE_FILE "${_ALPAKA_ROOT_DIR}/cmake/addExecutable.cmake")
INCLUDE("${_ALPAKA_ADD_EXECUTABLE_FILE}")

# Add ALPAKA_ADD_LIBRARY function.
SET(_ALPAKA_ADD_LIBRARY_FILE "${_ALPAKA_ROOT_DIR}/cmake/addLibrary.cmake")
INCLUDE("${_ALPAKA_ADD_LIBRARY_FILE}")

# Set found to true initially and set it to false if a required dependency is missing.
SET(_ALPAKA_FOUND TRUE)

# Add module search path
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${_ALPAKA_ROOT_DIR}/cmake/modules/")

#-------------------------------------------------------------------------------
# Options.
OPTION(ALPAKA_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode (This allows to mix alpaka code with native CUDA code)." OFF)

OPTION(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread back-end" ON)
OPTION(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" ON)
OPTION(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" ON)
OPTION(ALPAKA_ACC_CPU_BT_OMP4_ENABLE "Enable the OpenMP 4.0 CPU block and block thread back-end" OFF)
OPTION(ALPAKA_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" ON)

IF(ALPAKA_ACC_GPU_CUDA_ONLY_MODE AND
    (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OR
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
    ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
    ALPAKA_ACC_CPU_BT_OMP4_ENABLE))
    MESSAGE(WARNING "If ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled, only back-ends using CUDA can be enabled! This allows to mix alpaka code with native CUDA code. However, this prevents any non-CUDA back-ends from being enabled.")
ENDIF()

# Drop-down combo box in cmake-gui.
SET(ALPAKA_DEBUG "0" CACHE STRING "Debug level")
SET_PROPERTY(CACHE ALPAKA_DEBUG PROPERTY STRINGS "0;1;2")

#-------------------------------------------------------------------------------
# Debug output of common variables.
IF(${ALPAKA_DEBUG} GREATER 1)
    MESSAGE(STATUS "_ALPAKA_ROOT_DIR : ${_ALPAKA_ROOT_DIR}")
    MESSAGE(STATUS "_ALPAKA_COMMON_FILE : ${_ALPAKA_COMMON_FILE}")
    MESSAGE(STATUS "_ALPAKA_ADD_EXECUTABLE_FILE : ${_ALPAKA_ADD_EXECUTABLE_FILE}")
    MESSAGE(STATUS "_ALPAKA_ADD_LIBRARY_FILE : ${_ALPAKA_ADD_LIBRARY_FILE}")
    MESSAGE(STATUS "CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
ENDIF()

#-------------------------------------------------------------------------------
# Find Boost.
SET(_ALPAKA_BOOST_MIN_VER "1.62.0")
IF(${ALPAKA_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET COMPONENTS fiber context system thread chrono date_time)
    IF(NOT Boost_FIBER_FOUND)
        MESSAGE(STATUS "Optional alpaka dependency Boost fiber could not be found! Fibers back-end disabled!")
        SET(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OFF CACHE BOOL "Enable the Fibers CPU back-end" FORCE)
        FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET)
    ELSE()
        # On Win32 boost context triggers:
        # libboost_context-vc141-mt-gd-1_64.lib(jump_i386_ms_pe_masm.obj) : error LNK2026: module unsafe for SAFESEH image.
        IF(MSVC)
            IF(CMAKE_SIZEOF_VOID_P EQUAL 4)
                SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH:NO")
                SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO")
                SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /SAFESEH:NO")
            ENDIF()
        ENDIF()
    ENDIF()

ELSE()
    FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET)
ENDIF()

IF(${ALPAKA_DEBUG} GREATER 1)
    MESSAGE(STATUS "Boost in:")
    MESSAGE(STATUS "BOOST_ROOT : ${BOOST_ROOT}")
    MESSAGE(STATUS "BOOSTROOT : ${BOOSTROOT}")
    MESSAGE(STATUS "BOOST_INCLUDEDIR: ${BOOST_INCLUDEDIR}")
    MESSAGE(STATUS "BOOST_LIBRARYDIR: ${BOOST_LIBRARYDIR}")
    MESSAGE(STATUS "Boost_NO_SYSTEM_PATHS: ${Boost_NO_SYSTEM_PATHS}")
    MESSAGE(STATUS "Boost_ADDITIONAL_VERSIONS: ${Boost_ADDITIONAL_VERSIONS}")
    MESSAGE(STATUS "Boost_USE_MULTITHREADED: ${Boost_USE_MULTITHREADED}")
    MESSAGE(STATUS "Boost_USE_STATIC_LIBS: ${Boost_USE_STATIC_LIBS}")
    MESSAGE(STATUS "Boost_USE_STATIC_RUNTIME: ${Boost_USE_STATIC_RUNTIME}")
    MESSAGE(STATUS "Boost_USE_DEBUG_RUNTIME: ${Boost_USE_DEBUG_RUNTIME}")
    MESSAGE(STATUS "Boost_USE_DEBUG_PYTHON: ${Boost_USE_DEBUG_PYTHON}")
    MESSAGE(STATUS "Boost_USE_STLPORT: ${Boost_USE_STLPORT}")
    MESSAGE(STATUS "Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS: ${Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS}")
    MESSAGE(STATUS "Boost_COMPILER: ${Boost_COMPILER}")
    MESSAGE(STATUS "Boost_THREADAPI: ${Boost_THREADAPI}")
    MESSAGE(STATUS "Boost_NAMESPACE: ${Boost_NAMESPACE}")
    MESSAGE(STATUS "Boost_DEBUG: ${Boost_DEBUG}")
    MESSAGE(STATUS "Boost_DETAILED_FAILURE_MSG: ${Boost_DETAILED_FAILURE_MSG}")
    MESSAGE(STATUS "Boost_REALPATH: ${Boost_REALPATH}")
    MESSAGE(STATUS "Boost_NO_BOOST_CMAKE: ${Boost_NO_BOOST_CMAKE}")
    MESSAGE(STATUS "Boost out:")
    MESSAGE(STATUS "Boost_FOUND: ${Boost_FOUND}")
    MESSAGE(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    MESSAGE(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
    MESSAGE(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE(STATUS "Boost_FIBER_FOUND: ${Boost_FIBER_FOUND}")
    MESSAGE(STATUS "Boost_FIBER_LIBRARY: ${Boost_FIBER_LIBRARY}")
    MESSAGE(STATUS "Boost_CONTEXT_FOUND: ${Boost_CONTEXT_FOUND}")
    MESSAGE(STATUS "Boost_CONTEXT_LIBRARY: ${Boost_CONTEXT_LIBRARY}")
    MESSAGE(STATUS "Boost_SYSTEM_FOUND: ${Boost_SYSTEM_FOUND}")
    MESSAGE(STATUS "Boost_SYSTEM_LIBRARY: ${Boost_SYSTEM_LIBRARY}")
    MESSAGE(STATUS "Boost_THREAD_FOUND: ${Boost_THREAD_FOUND}")
    MESSAGE(STATUS "Boost_THREAD_LIBRARY: ${Boost_THREAD_LIBRARY}")
    MESSAGE(STATUS "Boost_ATOMIC_FOUND: ${Boost_ATOMIC_FOUND}")
    MESSAGE(STATUS "Boost_ATOMIC_LIBRARY: ${Boost_ATOMIC_LIBRARY}")
    MESSAGE(STATUS "Boost_CHRONO_FOUND: ${Boost_CHRONO_FOUND}")
    MESSAGE(STATUS "Boost_CHRONO_LIBRARY: ${Boost_CHRONO_LIBRARY}")
    MESSAGE(STATUS "Boost_DATE_TIME_FOUND: ${Boost_DATE_TIME_FOUND}")
    MESSAGE(STATUS "Boost_DATE_TIME_LIBRARY: ${Boost_DATE_TIME_LIBRARY}")
    MESSAGE(STATUS "Boost_VERSION: ${Boost_VERSION}")
    MESSAGE(STATUS "Boost_LIB_VERSION: ${Boost_LIB_VERSION}")
    MESSAGE(STATUS "Boost_MAJOR_VERSION: ${Boost_MAJOR_VERSION}")
    MESSAGE(STATUS "Boost_MINOR_VERSION: ${Boost_MINOR_VERSION}")
    MESSAGE(STATUS "Boost_SUBMINOR_VERSION: ${Boost_SUBMINOR_VERSION}")
    MESSAGE(STATUS "Boost_LIB_DIAGNOSTIC_DEFINITIONS: ${Boost_LIB_DIAGNOSTIC_DEFINITIONS}")
    MESSAGE(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE(STATUS "Boost cached:")
    MESSAGE(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
    MESSAGE(STATUS "Boost_LIBRARY_DIR: ${Boost_LIBRARY_DIR}")
ENDIF()

IF(NOT Boost_FOUND)
    MESSAGE(WARNING "Required alpaka dependency Boost (>=${_ALPAKA_BOOST_MIN_VER}) could not be found!")
    SET(_ALPAKA_FOUND FALSE)

ELSE()
    LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC ${Boost_INCLUDE_DIRS})
    LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC ${Boost_LIBRARIES})
ENDIF()

#-------------------------------------------------------------------------------
# Find TBB.
IF(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    FIND_PACKAGE(TBB 2.2)
    IF(NOT TBB_FOUND)
        MESSAGE(STATUS "Optional alpaka dependency TBB could not be found! TBB grid block back-end disabled!")
        SET(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE OFF CACHE BOOL "Enable the TBB grid block back-end" FORCE)
    ELSE()
        LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC ${TBB_LIBRARIES})
        LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC ${TBB_INCLUDE_DIRS})
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC ${TBB_DEFINITIONS})
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find OpenMP.
IF(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    FIND_PACKAGE(OpenMP)

    # Manually find OpenMP for the clang compiler if it was not already found.
    # Even CMake 3.5 is unable to find libiomp and provide the correct OpenMP flags.
    IF(NOT OPENMP_FOUND)
        IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            FIND_PATH(_ALPAKA_LIBIOMP_INCLUDE_DIR NAMES "omp.h" PATH_SUFFIXES "include" "libiomp" "include/libiomp")
            IF(_ALPAKA_LIBIOMP_INCLUDE_DIR)
                SET(OPENMP_FOUND TRUE)
                SET(OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
                SET(OpenMP_C_FLAGS "-fopenmp=libiomp5")
                LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC "${_ALPAKA_LIBIOMP_INCLUDE_DIR}")
            ENDIF()
        ENDIF()
    ENDIF()

    IF(NOT OPENMP_FOUND)
        MESSAGE(STATUS "Optional alpaka dependency OpenMP could not be found! OpenMP back-ends disabled!")
        SET(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU grid block back-end" FORCE)
        SET(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU block thread back-end" FORCE)
        SET(ALPAKA_ACC_CPU_BT_OMP4_ENABLE OFF CACHE BOOL "Enable the OpenMP 4.0 CPU block and thread back-end" FORCE)

    ELSE()
        # CUDA requires some special handling
        IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        ENDIF()

        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC ${OpenMP_CXX_FLAGS})
        IF(NOT MSVC)
            LIST(APPEND _ALPAKA_LINK_FLAGS_PUBLIC ${OpenMP_CXX_FLAGS})
        ENDIF()

        # clang versions beginning with 3.9 support OpenMP 4.0 but only when given the corresponding flag
        IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            IF(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
                LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-fopenmp-version=40")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-version=40")
            ENDIF()
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find CUDA.
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)

    IF(NOT DEFINED ALPAKA_CUDA_VERSION)
        SET(ALPAKA_CUDA_VERSION 7.0)
    ENDIF()

    IF(ALPAKA_CUDA_VERSION VERSION_LESS 7.0)
        MESSAGE(WARNING "CUDA Toolkit < 7.0 is not supported!")
        SET(_ALPAKA_FOUND FALSE)

    ELSE()
        FIND_PACKAGE(CUDA "${ALPAKA_CUDA_VERSION}")
        IF(NOT CUDA_FOUND)
            MESSAGE(STATUS "Optional alpaka dependency CUDA could not be found! CUDA back-end disabled!")
            SET(ALPAKA_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA GPU back-end" FORCE)

        ELSE()
            SET(ALPAKA_CUDA_VERSION "${CUDA_VERSION}")
            IF(CUDA_VERSION VERSION_LESS 9.0)
                SET(ALPAKA_CUDA_ARCH "20" CACHE STRING "GPU architecture")
            ELSE()
                SET(ALPAKA_CUDA_ARCH "30" CACHE STRING "GPU architecture")
            ENDIF()
            SET(ALPAKA_CUDA_COMPILER "nvcc" CACHE STRING "CUDA compiler")
            SET_PROPERTY(CACHE ALPAKA_CUDA_COMPILER PROPERTY STRINGS "nvcc;clang")

            OPTION(ALPAKA_CUDA_FAST_MATH "Enable fast-math" ON)
            OPTION(ALPAKA_CUDA_FTZ "Set flush to zero for GPU" OFF)
            OPTION(ALPAKA_CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
            OPTION(ALPAKA_CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)

            IF(ALPAKA_CUDA_COMPILER MATCHES "clang")
                FOREACH(_CUDA_ARCH_ELEM ${ALPAKA_CUDA_ARCH})
                    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "--cuda-gpu-arch=sm_${_CUDA_ARCH_ELEM}")
                ENDFOREACH()

                LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

                # This flag silences the warning produced by the Dummy.cpp files:
                # clang: warning: argument unused during compilation: '--cuda-gpu-arch=sm_XX'
                # This seems to be a false positive as all flags are 'unused' for an empty file.
                LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-Qunused-arguments")

                # Silences warnings that are produced by boost because clang is not correctly identified.
                LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-Wno-unused-local-typedef")

                IF(ALPAKA_CUDA_FAST_MATH)
                    # -ffp-contract=fast enables the usage of FMA
                    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-ffast-math" "-ffp-contract=fast")
                ENDIF()

                IF(ALPAKA_CUDA_FTZ)
                    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-fcuda-flush-denormals-to-zero")
                ENDIF()

                IF(ALPAKA_CUDA_SHOW_REGISTER)
                    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-Xcuda-ptxas=-v")
                ENDIF()

                IF(ALPAKA_CUDA_KEEP_FILES)
                    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-save-temps")
                ENDIF()

            ELSE()
                # Clean up the flags. Else, multiple find calls would result in duplicate flags. Furthermore, other modules may have set different settings.
                SET(CUDA_NVCC_FLAGS)

                IF(${ALPAKA_DEBUG} GREATER 1)
                    SET(CUDA_VERBOSE_BUILD ON)
                ENDIF()

                SET(CUDA_PROPAGATE_HOST_FLAGS ON)

                IF(CUDA_VERSION VERSION_EQUAL 8.0)
                    LIST(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
                ENDIF()

                IF(NOT CUDA_VERSION VERSION_LESS 7.5)
                    LIST(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
                    LIST(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
                ELSE()
                    # CUDA 7.0
                    LIST(APPEND CUDA_NVCC_FLAGS "--relaxed-constexpr")
                ENDIF()

                FOREACH(_CUDA_ARCH_ELEM ${ALPAKA_CUDA_ARCH})
                    # set flags to create device code for the given architecture
                    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
                        "--generate-code arch=compute_${_CUDA_ARCH_ELEM},code=sm_${_CUDA_ARCH_ELEM} --generate-code arch=compute_${_CUDA_ARCH_ELEM},code=compute_${_CUDA_ARCH_ELEM}")
                ENDFOREACH()

                IF(NOT MSVC)
                    LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
                ELSE()
                    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "_HAS_ITERATOR_DEBUGGING=0")
                ENDIF()

                SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

                IF(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
                    LIST(APPEND CUDA_NVCC_FLAGS "-g")
                    # https://github.com/ComputationalRadiationPhysics/alpaka/issues/428
                    IF(((CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0) OR
                        (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.8)) AND
                        CUDA_VERSION VERSION_LESS 9.0)
                        MESSAGE(WARNING "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} not support -G with CUDA <= 8! "
                                        "Device debug symbols NOT added.")
                    ELSE()
                        LIST(APPEND CUDA_NVCC_FLAGS "-G")
                    ENDIF()
                ENDIF()

                IF(ALPAKA_CUDA_FAST_MATH)
                    LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
                ENDIF()

                IF(ALPAKA_CUDA_FTZ)
                    LIST(APPEND CUDA_NVCC_FLAGS "--ftz=true")
                ELSE()
                    LIST(APPEND CUDA_NVCC_FLAGS "--ftz=false")
                ENDIF()

                IF(ALPAKA_CUDA_SHOW_REGISTER)
                    LIST(APPEND CUDA_NVCC_FLAGS "-Xptxas=-v")
                ENDIF()

                IF(ALPAKA_CUDA_KEEP_FILES)
                    MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
                    LIST(APPEND CUDA_NVCC_FLAGS "--keep" "--keep-dir" "${PROJECT_BINARY_DIR}/nvcc_tmp")
                ENDIF()

                OPTION(ALPAKA_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
                IF(ALPAKA_CUDA_SHOW_CODELINES)
                    LIST(APPEND CUDA_NVCC_FLAGS "--source-in-ptx" "-lineinfo")
                    IF(NOT MSVC)
                        LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-rdynamic")
                    ENDIF()
                    SET(ALPAKA_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
                ENDIF()
            ENDIF()

            LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;${CUDA_CUDART_LIBRARY}")
            LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC ${CUDA_INCLUDE_DIRS})
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Compiler settings.
IF(MSVC)
    # Empty append to define it if it does not already exist.
    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC)
ELSE()
    # Select C++ standard version.
    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-std=c++11")

    # Add linker options.
    # lipthread:
    LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;pthread")
    # librt: undefined reference to `clock_gettime'
    LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;rt")

    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-ftemplate-depth-512")
    # Clang or AppleClang
    ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-ftemplate-depth=512")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# alpaka.
IF(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_GPU_CUDA_ONLY_MODE")
    MESSAGE(STATUS ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
ENDIF()

IF(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_BT_OMP4_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_BT_OMP4_ENABLED)
ENDIF()
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_GPU_CUDA_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_GPU_CUDA_ENABLED)
ENDIF()

LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_DEBUG=${ALPAKA_DEBUG}")

IF(ALPAKA_CI)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_CI")
ENDIF()

SET(_ALPAKA_INCLUDE_DIRECTORY "${_ALPAKA_ROOT_DIR}/include")
LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC "${_ALPAKA_INCLUDE_DIRECTORY}")
SET(_ALPAKA_SUFFIXED_INCLUDE_DIR "${_ALPAKA_INCLUDE_DIRECTORY}/alpaka")

SET(_ALPAKA_LINK_LIBRARY)
LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "${_ALPAKA_LINK_LIBRARY}")

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" _ALPAKA_FILES_HEADER)
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "cpp" _ALPAKA_FILES_SOURCE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/script" "${_ALPAKA_ROOT_DIR}" "sh" _ALPAKA_FILES_SCRIPT)
SET_SOURCE_FILES_PROPERTIES(${_ALPAKA_FILES_SCRIPT} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/cmake" "${_ALPAKA_ROOT_DIR}" "cmake" _ALPAKA_FILES_CMAKE)
LIST(APPEND _ALPAKA_FILES_CMAKE "${_ALPAKA_ROOT_DIR}/alpakaConfig.cmake" "${_ALPAKA_ROOT_DIR}/Findalpaka.cmake" "${_ALPAKA_ROOT_DIR}/CMakeLists.txt" "${_ALPAKA_ROOT_DIR}/cmake/dev.cmake" "${_ALPAKA_ROOT_DIR}/cmake/common.cmake" "${_ALPAKA_ROOT_DIR}/cmake/addExecutable.cmake" "${_ALPAKA_ADD_LIBRRAY_FILE}")
SET_SOURCE_FILES_PROPERTIES(${_ALPAKA_FILES_CMAKE} PROPERTIES HEADER_FILE_ONLY TRUE)

append_recursive_files_add_to_src_group("${_ALPAKA_ROOT_DIR}/doc/markdown" "${_ALPAKA_ROOT_DIR}" "md" _ALPAKA_FILES_DOC)
SET_SOURCE_FILES_PROPERTIES(${_ALPAKA_FILES_DOC} PROPERTIES HEADER_FILE_ONLY TRUE)

SET(_ALPAKA_FILES_OTHER "${_ALPAKA_ROOT_DIR}/.gitignore" "${_ALPAKA_ROOT_DIR}/.travis.yml" "${_ALPAKA_ROOT_DIR}/appveyor.yml" "${_ALPAKA_ROOT_DIR}/COPYING" "${_ALPAKA_ROOT_DIR}/COPYING.LESSER" "${_ALPAKA_ROOT_DIR}/README.md")
SET_SOURCE_FILES_PROPERTIES(${_ALPAKA_FILES_OTHER} PROPERTIES HEADER_FILE_ONLY TRUE)

#-------------------------------------------------------------------------------
# Target.
IF(NOT TARGET "alpaka")
    ADD_LIBRARY(
        "alpaka"
        ${_ALPAKA_FILES_HEADER} ${_ALPAKA_FILES_SOURCE} ${_ALPAKA_FILES_SCRIPT} ${_ALPAKA_FILES_CMAKE} ${_ALPAKA_FILES_DOC} ${_ALPAKA_FILES_OTHER})

    # Compile options.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_COMPILE_OPTIONS_PUBLIC: ${_ALPAKA_COMPILE_OPTIONS_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_COMPILE_OPTIONS_PUBLIC
        _ALPAKA_COMPILE_OPTIONS_PUBLIC_LENGTH)
    IF(${_ALPAKA_COMPILE_OPTIONS_PUBLIC_LENGTH} GREATER 0)
        TARGET_COMPILE_OPTIONS(
            "alpaka"
            PUBLIC ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
    ENDIF()

    # Compile definitions.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_COMPILE_DEFINITIONS_PUBLIC: ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
    IF(${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC_LENGTH} GREATER 0)
        TARGET_COMPILE_DEFINITIONS(
            "alpaka"
            PUBLIC ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
    ENDIF()

    # Include directories.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC: ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC_LENGTH)
    IF(${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC_LENGTH} GREATER 0)
        TARGET_INCLUDE_DIRECTORIES(
            "alpaka"
            PUBLIC ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC})
    ENDIF()

    # Link libraries.
    # There are no PUBLIC_LINK_FLAGS in CMAKE:
    # http://stackoverflow.com/questions/26850889/cmake-keeping-link-flags-of-internal-libs
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_LINK_LIBRARIES_PUBLIC: ${_ALPAKA_LINK_LIBRARIES_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_LINK_LIBRARIES_PUBLIC
        _ALPAKA_LINK_LIBRARIES_PUBLIC_LENGTH)
    IF(${_ALPAKA_LINK_LIBRARIES_PUBLIC_LENGTH} GREATER 0)
        TARGET_LINK_LIBRARIES(
            "alpaka"
            PUBLIC ${_ALPAKA_LINK_LIBRARIES_PUBLIC} ${_ALPAKA_LINK_FLAGS_PUBLIC})
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find alpaka version.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_MAJOR_HPP REGEX "#define ALPAKA_VERSION_MAJOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_MINOR_HPP REGEX "#define ALPAKA_VERSION_MINOR ")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp" ALPAKA_VERSION_PATCH_HPP REGEX "#define ALPAKA_VERSION_PATCH ")

string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_MAJOR  ${ALPAKA_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_MINOR  ${ALPAKA_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" ALPAKA_VERSION_PATCH  ${ALPAKA_VERSION_PATCH_HPP})

SET(PACKAGE_VERSION "${ALPAKA_VERSION_MAJOR}.${ALPAKA_VERSION_MINOR}.${ALPAKA_VERSION_PATCH}")

#-------------------------------------------------------------------------------
# Set return values.
SET(alpaka_VERSION "${ALPAKA_VERSION_MAJOR}.${ALPAKA_VERSION_MINOR}.${ALPAKA_VERSION_PATCH}")
SET(alpaka_COMPILE_OPTIONS ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
SET(alpaka_COMPILE_DEFINITIONS ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
# Add '-D' to the definitions
SET(alpaka_DEFINITIONS ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
list_add_prefix("-D" alpaka_DEFINITIONS)
# Add the compile options to the definitions.
LIST(APPEND alpaka_DEFINITIONS ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
SET(alpaka_INCLUDE_DIR ${_ALPAKA_INCLUDE_DIRECTORY})
SET(alpaka_INCLUDE_DIRS ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC})
SET(alpaka_LIBRARY ${_ALPAKA_LINK_LIBRARY})
SET(alpaka_LIBRARIES ${_ALPAKA_LINK_FLAGS_PUBLIC})
LIST(APPEND alpaka_LIBRARIES ${_ALPAKA_LINK_LIBRARIES_PUBLIC})

#-------------------------------------------------------------------------------
# Print the return values.
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "alpaka_FOUND: ${alpaka_FOUND}")
    MESSAGE(STATUS "alpaka_VERSION: ${alpaka_VERSION}")
    MESSAGE(STATUS "alpaka_COMPILE_OPTIONS: ${alpaka_COMPILE_OPTIONS}")
    MESSAGE(STATUS "alpaka_COMPILE_DEFINITIONS: ${alpaka_COMPILE_DEFINITIONS}")
    MESSAGE(STATUS "alpaka_DEFINITIONS: ${alpaka_DEFINITIONS}")
    MESSAGE(STATUS "alpaka_INCLUDE_DIR: ${alpaka_INCLUDE_DIR}")
    MESSAGE(STATUS "alpaka_INCLUDE_DIRS: ${alpaka_INCLUDE_DIRS}")
    MESSAGE(STATUS "alpaka_LIBRARY: ${alpaka_LIBRARY}")
    MESSAGE(STATUS "alpaka_LIBRARIES: ${alpaka_LIBRARIES}")
ENDIF()

# Unset already set variables if not found.
IF(NOT _ALPAKA_FOUND)
    UNSET(alpaka_FOUND)
    UNSET(alpaka_VERSION)
    UNSET(alpaka_COMPILE_OPTIONS)
    UNSET(alpaka_COMPILE_DEFINITIONS)
    UNSET(alpaka_DEFINITIONS)
    UNSET(alpaka_INCLUDE_DIR)
    UNSET(alpaka_INCLUDE_DIRS)
    UNSET(alpaka_LIBRARY)
    UNSET(alpaka_LIBRARIES)

    UNSET(_ALPAKA_FOUND)
    UNSET(_ALPAKA_COMPILE_OPTIONS_PUBLIC)
    UNSET(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC)
    UNSET(_ALPAKA_INCLUDE_DIRECTORY)
    UNSET(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC)
    UNSET(_ALPAKA_LINK_LIBRARY)
    UNSET(_ALPAKA_LINK_LIBRARIES_PUBLIC)
    UNSET(_ALPAKA_LINK_FLAGS_PUBLIC)
    UNSET(_ALPAKA_COMMON_FILE)
    UNSET(_ALPAKA_ADD_EXECUTABLE_FILE)
    UNSET(_ALPAKA_ADD_LIBRARY_FILE)
    UNSET(_ALPAKA_FILES_HEADER)
    UNSET(_ALPAKA_FILES_SOURCE)
    UNSET(_ALPAKA_FILES_OTHER)
    UNSET(_ALPAKA_BOOST_MIN_VER)
    UNSET(_ALPAKA_VERSION_DEFINE)
    UNSET(_ALPAKA_VER_MAJOR)
    UNSET(_ALPAKA_VER_MINOR)
    UNSET(_ALPAKA_VER_PATCH)
ELSE()
    # Make internal variables advanced options in the GUI.
    MARK_AS_ADVANCED(
        alpaka_INCLUDE_DIR
        alpaka_LIBRARY
        _ALPAKA_COMPILE_OPTIONS_PUBLIC
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC
        _ALPAKA_INCLUDE_DIRECTORY
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC
        _ALPAKA_LINK_LIBRARY
        _ALPAKA_LINK_LIBRARIES_PUBLIC
        _ALPAKA_LINK_FLAGS_PUBLIC
        _ALPAKA_COMMON_FILE
        _ALPAKA_ADD_EXECUTABLE_FILE
        _ALPAKA_ADD_LIBRARY_FILE
        _ALPAKA_FILES_HEADER
        _ALPAKA_FILES_SOURCE
        _ALPAKA_FILES_OTHER
        _ALPAKA_BOOST_MIN_VER
        _ALPAKA_VERSION_DEFINE
        _ALPAKA_VER_MAJOR
        _ALPAKA_VER_MINOR
        _ALPAKA_VER_PATCH)
ENDIF()

###############################################################################
# FindPackage options

# Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
# NOTE: We do not check for alpaka_LIBRARIES and alpaka_DEFINITIONS because they can be empty.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    "alpaka"
    FOUND_VAR alpaka_FOUND
    REQUIRED_VARS alpaka_INCLUDE_DIR
    VERSION_VAR alpaka_VERSION)
