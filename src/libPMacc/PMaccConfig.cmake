# Copyright 2015-2017 Erik Zenker, Rene Widera, Axel Huebl
#
# This file is part of libPMacc.
#
# libPMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libPMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with libPMacc.
# If not, see <http://www.gnu.org/licenses/>.
#


# - Config file for the pmacc package
# It defines the following variables
#  PMacc_INCLUDE_DIRS - include directories for pmacc
#  PMacc_LIBRARIES    - libraries to link against
#  PMacc_DEFINITIONS  - definitions of pmacc

###############################################################################
# PMacc
###############################################################################
cmake_minimum_required(VERSION 3.3.0)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

# own modules for find_packages e.g. FindmallocMC
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PMacc_DIR}/../../thirdParty/cmake-modules/)


###############################################################################
# Build Flags
###############################################################################

set(PMACC_BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type for the project" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${PMACC_BUILD_TYPE}")
unset(PMACC_BUILD_TYPE)


###############################################################################
# Language Flags
###############################################################################

# enforce C++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)


###############################################################################
# Select CUDA Compiler
###############################################################################

set(PMACC_DEVICE_COMPILER "nvcc")
set(PMACC_CUDA_COMPILER "${PMACC_DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(APPEND PMACC_DEVICE_COMPILER "clang")
endif()

set_PROPERTY(CACHE PMACC_CUDA_COMPILER PROPERTY STRINGS "${PMACC_DEVICE_COMPILER}")


################################################################################
# Find CUDA
################################################################################

find_package(CUDA 7.5 REQUIRED)

set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${CUDA_LIBRARIES})

set(CUDA_ARCH "20" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

foreach(PMACC_CUDA_ARCH_ELEM ${CUDA_ARCH})
    string(REGEX MATCH "^[0-9]+$" PMACC_IS_NUMBER ${CUDA_ARCH})
    if(NOT PMACC_IS_NUMBER)
        message(FATAL_ERROR "Defined compute architecture '${PMACC_CUDA_ARCH_ELEM}' in "
                            "'${CUDA_ARCH}' is not an integral number, use e.g. '20' (for SM 2.0).")
    endif()
    unset(PMACC_IS_NUMBER)

    if(${PMACC_CUDA_ARCH_ELEM} LESS 20)
        message(FATAL_ERROR "Unsupported CUDA architecture '${PMACC_CUDA_ARCH_ELEM}' specified. "
                            "Use '20' (for SM 2.0) or higher.")
    endif()
endforeach()

option(CUDA_FAST_MATH "Enable fast-math" ON)
option(CUDA_FTZ "Set flush to zero for CUDA" OFF)
option(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)

if(CUDA_KEEP_FILES)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/nvcc_tmp")
endif(CUDA_KEEP_FILES)

if("${PMACC_CUDA_COMPILER}" STREQUAL "clang")
    add_definitions(-DPMACC_CUDA_COMPILER_CLANG=1)
    set(CLANG_BUILD_FLAGS "-O3 -x cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    # activation usage of FMA
    set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -ffp-contract=fast")

    if(CUDA_FAST_MATH)
        # enable fast math
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -ffast-math")
    endif()

    if(CUDA_FTZ)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -fcuda-flush-denormals-to-zero")
    else()
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -fno-cuda-flush-denormals-to-zero")
    endif()

    if(CUDA_SHOW_REGISTER)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -Xcuda-ptxas -v")
    endif(CUDA_SHOW_REGISTER)

    if(CUDA_KEEP_FILES)
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} -save-temps=${PROJECT_BINARY_DIR}/nvcc_tmp")
    endif(CUDA_KEEP_FILES)

    foreach(PMACC_CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architectures
        set(CLANG_BUILD_FLAGS "${CLANG_BUILD_FLAGS} --cuda-gpu-arch=sm_${PMACC_CUDA_ARCH_ELEM}")
    endforeach()

elseif("${PMACC_CUDA_COMPILER}" STREQUAL "nvcc")
    add_definitions(-DPMACC_CUDA_COMPILER_CLANG=0)
    # work-around since the above flag does not necessarily put -std=c++11 in
    # the CMAKE_CXX_FLAGS, which is unfurtunately hiding the switch from FindCUDA
    if(NOT "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+11")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()

    foreach(PMACC_CUDA_ARCH_ELEM ${CUDA_ARCH})
        # set flags to create device code for the given architecture
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            "--generate-code arch=compute_${PMACC_CUDA_ARCH_ELEM},code=sm_${PMACC_CUDA_ARCH_ELEM} --generate-code arch=compute_${PMACC_CUDA_ARCH_ELEM},code=compute_${PMACC_CUDA_ARCH_ELEM}")
    endforeach()

    if(CUDA_FAST_MATH)
        # enable fast math
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
    endif()

    if(CUDA_FTZ)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --ftz=true)
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --ftz=false)
    endif()

    option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

    if(CUDA_SHOW_CODELINES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -Xcompiler -rdynamic -lineinfo)
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    if(CUDA_SHOW_REGISTER)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
    endif(CUDA_SHOW_REGISTER)

    if(CUDA_KEEP_FILES)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
    endif(CUDA_KEEP_FILES)

else()
    message(FATAL_ERROR "selected CUDA compiler '${PMACC_CUDA_COMPILER}' is not supported")
endif()


################################################################################
# VampirTrace
################################################################################

option(VAMPIR_ENABLE "Create PMacc with VampirTrace support" OFF)

# set filters: please do NOT use line breaks WITHIN the string!
set(VT_INST_FILE_FILTER
    "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp"
    CACHE STRING "VampirTrace: Files to exclude from instrumentation")
set(VT_INST_FUNC_FILTER
    "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
    CACHE STRING "VampirTrace: Functions to exclude from instrumentation")

if(VAMPIR_ENABLE)
    message(STATUS "Building with VampirTrace support")
    set(VAMPIR_ROOT "$ENV{VT_ROOT}")
    if(NOT VAMPIR_ROOT)
        message(FATAL_ERROR "Environment variable VT_ROOT not set!")
    endif(NOT VAMPIR_ROOT)

    # compile flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
                    OUTPUT_VARIABLE VT_COMPILEFLAGS
                    RESULT_VARIABLE VT_CONFIG_RETURN
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT VT_CONFIG_RETURN EQUAL 0)
        message(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check file permissions")
    endif()
    # link flags
    execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
                    OUTPUT_VARIABLE VT_LINKFLAGS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    # bugfix showme
    string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")

    # modify our flags
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${VT_LINKFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VT_COMPILEFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${VT_INST_FILE_FILTER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")

    # nvcc flags (rly necessary?)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${VT_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
        -v)

    # for manual instrumentation and hints that vampir is enabled in our code
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DVTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DMPICH_IGNORE_CXX_SEEK)
endif(VAMPIR_ENABLE)


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_C_LIBRARIES})

# bullxmpi fails if it can not find its c++ counter part
if(MPI_CXX_FOUND)
    set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.57.0 REQUIRED COMPONENTS filesystem system math_tr1)
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${Boost_LIBRARIES})

# Boost 1.55 added support for a define that makes result_of look for
# the result<> template and falls back to decltype if none is found. This is
# great for the transition from the "wrong" usage to the "correct" one as
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DBOOST_RESULT_OF_USE_TR1_WITH_DECLTYPE_FALLBACK)

# Boost >= 1.60.0 and CUDA != 7.5 failed when used with C++11
# seen with boost 1.60.0 - 1.62.0 (atm latest) and CUDA 7.0, 8.0 (atm latest)
# CUDA 7.5 works without a workaround
if( (Boost_VERSION GREATER 105999) AND
    (NOT CUDA_VERSION VERSION_EQUAL 7.5) )
    # Boost Bug https://svn.boost.org/trac/boost/ticket/11897
    message(STATUS "Boost: Disable template aliases")
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} -DBOOST_NO_CXX11_TEMPLATE_ALIASES)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # suppress boost error
    # 'no member named "impl" in "boost::detail::thread_move_t<boost::detail::nullary_function<void ()> >"'
    # in 'boost/thread/detail/nullary_function.hpp'
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SMART_PTR")
endif()

# Boost 1.64.0 is broken with CUDA 8.0 (nvcc) and C++11
#   https://github.com/ComputationalRadiationPhysics/picongpu/issues/2048
#   fixed in CUDA 9.0 (ticket 1928813)
if( ("${PMACC_CUDA_COMPILER}" STREQUAL "nvcc") AND
    (Boost_VERSION EQUAL 106400) AND
    (CUDA_VERSION VERSION_EQUAL 8.0) )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_NOEXCEPT")
endif()


################################################################################
# Find OpenMP
################################################################################

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


################################################################################
# Find mallocMC
################################################################################

find_package(mallocMC 2.2.0 QUIET)

if(NOT mallocMC_FOUND)
    message(STATUS "Using mallocMC from thirdParty/ directory")
    set(MALLOCMC_ROOT "${PMacc_DIR}/../../thirdParty/mallocMC")
    find_package(mallocMC 2.2.0 REQUIRED)
endif(NOT mallocMC_FOUND)

set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${mallocMC_INCLUDE_DIRS})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${mallocMC_LIBRARIES})
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} ${mallocMC_DEFINITIONS})


################################################################################
# PMacc options
################################################################################

option(PMACC_BLOCKING_KERNEL
    "activate checks for every kernel call and synch after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_SYNC_KERNEL=1")
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for libPMacc")
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE}")

# PMacc header files
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} "${PMacc_DIR}/include")
