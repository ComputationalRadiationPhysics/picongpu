# Copyright 2015 Erik Zenker, Rene Widera, Axel Huebl
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
# pmacc
###############################################################################
cmake_minimum_required(VERSION 3.3)
project("PMacc")

set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} "${PMacc_DIR}/include")

# This path should be within PMacc
# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${PMacc_DIR}/../../thirdParty/cmake-modules/)

# Options
option(PMACC_BLOCKING_KERNEL "activate checks for every kernel call and synch after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
  set(pmacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_SYNC_KERNEL=1")
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for libPMacc")
set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} "-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE}")

###############################################################################
# Compiler Flags
###############################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()

################################################################################
# CUDA
################################################################################
find_package(CUDA 5.0 REQUIRED)

if(CUDA_VERSION VERSION_LESS 5.5)
    message(STATUS "CUDA Toolkit < 5.5 detected. We strongly recommend to still "
                   "use CUDA 5.5+ drivers (319.82 or higher)!")
endif(CUDA_VERSION VERSION_LESS 5.5)

set(CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
string(COMPARE EQUAL ${CUDA_ARCH} "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
string(COMPARE EQUAL ${CUDA_ARCH} "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
string(COMPARE EQUAL ${CUDA_ARCH} "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
string(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

if(IS_CUDA_ARCH_UNSUPPORTED)
    message(FATAL_ERROR "Unsupported CUDA architecture ${CUDA_ARCH} specified. "
                       "SM 2.0 or higher is required.")
endif(IS_CUDA_ARCH_UNSUPPORTED)

set(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")

set(CUDA_MATH --use_fast_math CACHE STRING "Enable fast-math" )
option(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

if(CUDA_SHOW_CODELINES)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -Xcompiler -rdynamic -lineinfo)
    set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
endif(CUDA_SHOW_CODELINES)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
if(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
endif(CUDA_SHOW_REGISTER)

if(CUDA_KEEP_FILES)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
endif(CUDA_KEEP_FILES)

################################################################################
# VAMPIR
################################################################################
option(VAMPIR_ENABLE "create PMacc vampir support" OFF)

if(VAMPIR_ENABLE)
  message("[CONFIG]  build program with vampir support")
  set(CMAKE_CXX_COMPILER "vtc++")
  set(CMAKE_CXX_INST_FILE_FILTER "stl,usr/include,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandardAccessor.hpp,StandardNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp")
  set(CMAKE_CXX_INST_FUNC_FILTER "vector,Vector,dim3,PMacc,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance")
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")

  # nvcc flags (rly necessary?)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=stl,-finstrument-functions-exclude-function-list='GPUGrid,execute,allocator,Task,Manager,Transaction,Mask',-pthread)

  set(LIBS vt-hyb ${LIBS})
endif(VAMPIR_ENABLE)

################################################################################
# MPI LIB
################################################################################
find_package(MPI MODULE REQUIRED)
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_C_LIBRARIES})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${MPI_CXX_LIBRARIES})

################################################################################
# PNGwriter
################################################################################
find_package(PNGwriter MODULE REQUIRED)

if(PNGwriter_FOUND)
  include_directories( )
  set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${PNGwriter_INCLUDE_DIRS})
  list(APPEND PNGwriter_DEFINITIONS "-DGOL_ENABLE_PNG=1")
  set(PMacc_DEFINITIONS ${PMacc_DEFINITIONS} ${PNGwriter_DEFINITIONS})
  set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${PNGwriter_LIBRARIES})
endif(PNGwriter_FOUND)

###############################################################################
# Boost LIB
###############################################################################
find_package(Boost 1.56.0 MODULE REQUIRED COMPONENTS program_options regex system)
set(PMacc_INCLUDE_DIRS ${PMacc_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${Boost_LIBRARIES})

################################################################################
# PThreads
################################################################################
find_package(Threads MODULE REQUIRED)
set(PMacc_LIBRARIES ${PMacc_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
