# Copyright 2013-2015 Rene Widera, Axel Huebl
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with libPMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.12.2)


################################################################################
# Project
################################################################################

project(GameOfLife)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif()

# set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{VT_ROOT}")

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../thirdParty/cmake-modules/)


################################################################################
# Find CUDA
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
# VampirTrace
################################################################################

option(VAMPIR_ENABLE "create gameOfLife with vampir support" OFF)

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
# Build type (debug, release)
################################################################################

option(GOL_RELEASE "Build release version, disables all runtime asserts" OFF)
if(GOL_RELEASE)
    message(STATUS "Release version")

    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
else(GOL_RELEASE)
    message(STATUS "Debug version")

    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-g" "-Xcompiler=-g,-pthread")
endif(GOL_RELEASE)


################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})

# bullxmpi fails if it can not find its c++ counter part
if(MPI_CXX_FOUND)
    set(LIBS ${LIBS} ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)


################################################################################
# Find PThreads
################################################################################

find_package(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})


###############################################################################
# Find Boost
###############################################################################

find_package(Boost 1.49.0 REQUIRED COMPONENTS program_options regex system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# work-arounds
if( (Boost_VERSION EQUAL 105500) AND
    (CUDA_VERSION VERSION_LESS 6.5) )
    # Boost Bug https://svn.boost.org/trac/boost/ticket/9392
    # nvbug #1422182 submission ID #391854
    message(STATUS "Boost: Applying noinline work around")
    set(CUDA_NVCC_FLAGS
      "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif()


################################################################################
# PMacc options
################################################################################

option(PMACC_BLOCKING_KERNEL "Activate checks for every kernel call and synchronize after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    add_definitions(-DPMACC_SYNC_KERNEL=1)
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "Set verbosity level for libPMacc")
add_definitions(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})


################################################################################
# libPMacc include directories
################################################################################

find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../libPMacc"
  DOC "libPMacc root location."
  )

include_directories(${PMACC_ROOT_DIR}/include)


################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.5.5 REQUIRED)

if(PNGwriter_FOUND)
    include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
    list(APPEND PNGwriter_DEFINITIONS "-DGOL_ENABLE_PNG=1")
    add_definitions(${PNGwriter_DEFINITIONS})
    set(LIBS ${LIBS} ${PNGwriter_LIBRARIES})
endif(PNGwriter_FOUND)


################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
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
# Compile & Link GoL
################################################################################

include_directories(include)

file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(gameOfLife
    ${CUDASRCFILES}
    ${SRCFILES}
)

target_link_libraries(gameOfLife  ${LIBS} ${CUDA_CUDART_LIBRARY} z m ${MPI_EXTRA_LIBRARY})


################################################################################
# Install GoL
################################################################################

install(TARGETS gameOfLife
        RUNTIME DESTINATION bin)

set(GOL_COPY_ON_INSTALL "submit" CACHE LIST "folder which copied on install to install path" )

# copy all subfolders (defined in: GOL_COPY_ON_INSTALL) to install folder
foreach(dir ${GOL_COPY_ON_INSTALL})
  #if source not exists than copy
  if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
      # copy important subfolders from extension path (default picongpu parameter)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/")
        INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
          DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
          PATTERN .svn EXCLUDE
        )
      endif()
  endif()
endforeach()
