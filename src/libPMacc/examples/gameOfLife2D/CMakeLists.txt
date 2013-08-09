#
# Copyright 2013 René Widera
#
# This file is part of libPMacc. 
# 
# libPMacc is free software: you can redistribute it and/or modify 
# it under the terms of of either the GNU General Public License or 
# the GNU Lesser General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 
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

cmake_minimum_required(VERSION 2.6)

OPTION(VAMPIR_ENABLE "create gameOfLife with vampir support" OFF)

IF(VAMPIR_ENABLE)
    message("[CONFIG]  build program with vampir support")
    set(CMAKE_CXX_COMPILER "vtc++")
    set(CMAKE_CXX_INST_FILE_FILTER "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandartAccessor.hpp,StandartNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp")
    set(CMAKE_CXX_INST_FUNC_FILTER "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
ENDIF(VAMPIR_ENABLE)

# Projekt name
project (GameOfLife)

#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

FIND_PACKAGE(CUDA REQUIRED)

OPTION(GOL_RELEASE "disable all runtime asserts" OFF)
IF(GOL_RELEASE)
    SET(CMAKE_BUILD_TYPE Release)
    ADD_DEFINITIONS(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
ELSE(GOL_RELEASE)
    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-g" "-Xcompiler=-g,-pthread")
ENDIF(GOL_RELEASE)
SET(CMAKE_CXX_FLAGS_DEFAULT "-Wall")

FIND_PACKAGE(MPI REQUIRED)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
SET(LIBS ${LIBS}${MPI_LIBRARIES}) 

########################################################################
#set options

##CUDA##
SET(CUDA_ARCH sm_13 CACHE STRING "set GPU architecture" )
STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_SM13)
SET(CUDA_FTZ "--ftz=false" CACHE STRING "set flush to zero for GPU")
IF( IS_CUDA_ARCH_SM13)
    SET(CUDA_FTZ "")
ENDIF( IS_CUDA_ARCH_SM13)

SET(CUDA_MATH --use_fast_math CACHE STRING "use intrinsic GPU math functions" )
OPTION(CUDA_SHOW_REGISTER "show kernel register and create PTX" OFF)
OPTION(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps. (folder: nvcc_tmp)" OFF)


SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
IF(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v -Xopencc=-LIST:source=on)
ENDIF(CUDA_SHOW_REGISTER)

IF(CUDA_KEEP_FILES)
    MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "nvcc_tmp")
ENDIF(CUDA_KEEP_FILES)

##END CUDA##


OPTION(GOL_ENABLE_PNG "create gameOfLife png output support (need pngwriter)" ON)
IF(GOL_ENABLE_PNG)
    ADD_DEFINITIONS(-DGOL_ENABLE_PNG=1)
ENDIF(GOL_ENABLE_PNG)

#########################################################################
# Configure include directories

#LIBGPUGRID

  find_path(GOL_LIBGPUGRID_ROOT_DIR
    NAMES include/types.h
    PATHS "${CMAKE_SOURCE_DIR}/../../../libgpugrid"
    DOC "libgpugrid root location."
    )

INCLUDE_DIRECTORIES(${GOL_LIBGPUGRID_ROOT_DIR}/include)

#PNGWRITER
IF(GOL_ENABLE_PNG)
  find_path(GOL_PNGWRITER_ROOT_DIR
    NAMES include/pngwriter.h
    PATHS ENV PNGWRITER_ROOT 
    DOC "pngwriter root location. (only used if GOL_ENABLE_PNG is enabled)"
    )
ENDIF(GOL_ENABLE_PNG)
IF(GOL_ENABLE_PNG)
    INCLUDE_DIRECTORIES(${GOL_PNGWRITER_ROOT_DIR}/include /usr/include/freetype2/)
    LINK_DIRECTORIES(${GOL_PNGWRITER_ROOT_DIR}/lib)
    SET(LIBS ${LIBS} png pngwriter freetype)
ENDIF(GOL_ENABLE_PNG)

#Boost from system
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex)
INCLUDE_DIRECTORIES(AFTER ${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
SET(LIBS ${LIBS} ${Boost_LIBRARIES})


################################################################################
INCLUDE_DIRECTORIES(include ${MPI_INCLUDE_PATH} ${mpi_include_path})
################################################################################


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${INCLUDE_DIRECTORIES})

IF(VAMPIR_ENABLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=stl,-finstrument-functions-exclude-function-list='GPUGrid,execute,allocator,Task,Manager,Transaction,Mask',-pthread )
ENDIF(VAMPIR_ENABLE)


file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(gameOfLife
    ${CUDASRCFILES}
    ${SRCFILES}
)

IF(VAMPIR_ENABLE)
    SET(LIBS vt-hyb ${LIBS} )
ENDIF(VAMPIR_ENABLE)

target_link_libraries(gameOfLife  ${LIBS} ${CUDA_CUDART_LIBRARY} z m ${MPI_EXTRA_LIBRARY})


###install section###
INSTALL(TARGETS gameOfLife
         RUNTIME DESTINATION bin)


