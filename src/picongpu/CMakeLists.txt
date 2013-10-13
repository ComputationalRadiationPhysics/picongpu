#
# Copyright 2013 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau, René Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.5)


################################################################################
# Pre-project (any compiler must be set here)
################################################################################

option(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)
if(VAMPIR_ENABLE)
    message(STATUS "Building with VampirTrace support")
    set(CMAKE_CXX_COMPILER "vtc++")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib
        -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER}
        -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
    set(CMAKE_CXX_FLAGS_DEFAULT "${CMAKE_CXX_FLAGS_DEFAULT} -vt:hyb -L/$ENV{VT_ROOT}/lib
        -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER}
        -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
endif(VAMPIR_ENABLE)


################################################################################
# Project 
################################################################################

project(PIConGPU)

# set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}" "$ENV{VT_ROOT}")


################################################################################
# Find CUDA 
################################################################################

find_package(CUDA REQUIRED)

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
    make_directory("${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
endif(CUDA_KEEP_FILES)


################################################################################
# Build type (debug, release)
################################################################################

option(PIC_RELEASE "Build release version, disables all runtime asserts" OFF)
if(PIC_RELEASE)
    message(STATUS "Release version")

    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
else(PIC_RELEASE)
    message(STATUS "Debug version")

    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g,-pthread")
endif(PIC_RELEASE)


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


################################################################################
# Find Boost
################################################################################

find_package(Boost REQUIRED COMPONENTS program_options regex)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 


################################################################################
# PMacc options
################################################################################

option(PMACC_BLOCKING_KERNEL 
       "Activate checks for every kernel call and synchronize after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    add_definitions(-DPMACC_SYNC_KERNEL=1)
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "Set verbosity level for libPMacc")
add_definitions(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})


################################################################################
# PARAM (overwrite) defines
################################################################################

if(PARAM_OVERWRITES)
    foreach(param ${PARAM_OVERWRITES})
       add_definitions(${param})
    endforeach(param)
endif(PARAM_OVERWRITES)


################################################################################
# load cuda_memtest project
################################################################################

set(SAME_NVCC_FLAGS_IN_SUBPROJECTS OFF)
find_path(CUDA_MEMTEST_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../cuda_memtest"
        DOC "path to cuda_memtest"
        )

add_subdirectory(${CUDA_MEMTEST_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_cuda_memtest")


################################################################################
# Load mpiInfo project
################################################################################

find_path(MPI_INFO_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../mpiInfo"
        DOC "path to mpiInfo"
        )

add_subdirectory(${MPI_INFO_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_mpiInfo")


################################################################################
# PIConGPU options
################################################################################

set(PIC_VERBOSE "1" CACHE STRING
    "Set verbosity level for PIConGPU (default is only physics output)")
add_definitions(-DPIC_VERBOSE_LVL=${PIC_VERBOSE})

option(PIC_ENABLE_INSITU_VOLVIS "Enable In Situ Volume Visualization" OFF)
if(PIC_ENABLE_INSITU_VOLVIS)
	add_definitions(-DENABLE_INSITU_VOLVIS=1)
	set(CUDA_ARCH sm_35 CACHE STRING "set GPU architecture")
	set(IceT_DIR "$ENV{ICET_ROOT}/lib/")
	find_package(IceT REQUIRED)
	include_directories(SYSTEM ${ICET_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${ICET_CORE_LIBS} ${ICET_MPI_LIBS})
endif(PIC_ENABLE_INSITU_VOLVIS)


################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()


################################################################################
# Configure include directories
################################################################################

# libPMacc

find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../libPMacc"
  DOC "libPMacc root location"
  )
include_directories(${PMACC_ROOT_DIR}/include)


################################################################################
# libSplash (+ hdf5 due to required headers)
################################################################################

# find libSplash installation
find_path(PIC_SPLASH_ROOT_DIR
  NAMES include/splash.h lib/libsplash.a
  PATHS ENV SPLASH_ROOT
  DOC "libSplash ROOT location (provides HDF5 output)"
)

if(PIC_SPLASH_ROOT_DIR)
    message(STATUS "Found libSplash: "${PIC_SPLASH_ROOT_DIR})
    find_package(HDF5 REQUIRED)

    # libSplash compiled with parallel support?
    file(STRINGS "${PIC_SPLASH_ROOT_DIR}/include/splash.h" _splash_H_CONTENTS
         REGEX "#define SPLASH_SUPPORTED_PARALLEL ")
    string(REGEX MATCH "([0-9]+)" _splash_IS_PARALLEL "${_splash_H_CONTENTS}")

    # check parallel libSplash to parallel hdf5 compatibility
    if("${_splash_IS_PARALLEL}")
       message(STATUS "libSplash supports PARALLEL output")
       if(NOT HDF5_IS_PARALLEL)
           message(FATAL_ERROR "libSplash compiled with PARALLEL support but HDF5 lacks it...")
       endif(NOT HDF5_IS_PARALLEL)
    endif("${_splash_IS_PARALLEL}")

    add_definitions(-DENABLE_HDF5=1)

    # add include information
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
    include_directories(SYSTEM ${PIC_SPLASH_ROOT_DIR}/include)

    # add library information
    # dynamic / shared
    #link_directories("${PIC_SPLASH_ROOT_DIR}/lib")   
    #set(LIBS ${LIBS} splash ${HDF5_LIBRARIES})
    #target_link_libraries(splash hdf5)

    # static
    add_library(splash_static STATIC IMPORTED)
    set_target_properties(splash_static PROPERTIES IMPORTED_LOCATION
                          ${PIC_SPLASH_ROOT_DIR}/lib/libsplash.a)
    set(LIBS ${LIBS} splash_static ${HDF5_LIBRARIES})
else(PIC_SPLASH_ROOT_DIR)
    message(STATUS "Could NOT find libSplash for hdf5 output")
endif(PIC_SPLASH_ROOT_DIR)


################################################################################
# PNGWRITER
################################################################################

find_path(PIC_PNGWRITER_ROOT_DIR
    NAMES include/pngwriter.h
    PATHS ENV PNGWRITER_ROOT 
    DOC "pngwriter ROOT location")

if(PIC_PNGWRITER_ROOT_DIR)
    message(STATUS "Found PNGWriter: "${PIC_PNGWRITER_ROOT_DIR})

    add_definitions(-DPIC_ENABLE_PNG=1)
    include_directories(SYSTEM ${PIC_PNGWRITER_ROOT_DIR}/include)
    link_directories(${PIC_PNGWRITER_ROOT_DIR}/lib)
    set(LIBS ${LIBS} png pngwriter)

    # freetype support enabled?
    include(FindFreetype)
    if(FREETYPE_FOUND)
      include_directories(${FREETYPE_INCLUDE_DIRS})
      set(LIBS ${LIBS} ${FREETYPE_LIBRARIES})
    endif(FREETYPE_FOUND)
else(PIC_PNGWRITER_ROOT_DIR)
    message(STATUS "Could NOT find PNGWriter")
endif(PIC_PNGWRITER_ROOT_DIR)


################################################################################
# Check if PIC_EXTENSION_PATH is relative or absolute
################################################################################

find_path(PIC_EXTENSION_PATH
    NAMES include/simulation_defines/param/componentsConfig.param
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
    DOC "search PATH to extension folder"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )

set(PIC_COPY_ON_INSTALL "include/simulation_defines" "submit" CACHE LIST
    "Folder which is copied to installation path during install" )

include_directories(include)
include_directories(BEFORE ${PIC_EXTENSION_PATH}/include)


################################################################################
# VampirTrace support
################################################################################

if(VAMPIR_ENABLE)
    find_path(PIC_VT_ROOT_DIR
        NAMES include/vampirtrace/vt_user.h lib/libvt-hyb.so
        PATHS ENV VT_ROOT 
        DOC "VampirTrace ROOT location")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${CMAKE_CXX_INST_FILE_FILTER}\\\"
        -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${CMAKE_CXX_INST_FUNC_FILTER}\\\"
        -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${PIC_VT_ROOT_DIR}/include/vampirtrace\\\"
        -v)
    set(LIBS vt-hyb ${LIBS})
endif(VAMPIR_ENABLE)


################################################################################
# Compile & Link PIConGPU
################################################################################

file(GLOB CUDASRCFILES "*.cu")
file(GLOB SRCFILES "*.cpp")

cuda_add_executable(picongpu
    ${CUDASRCFILES}
    ${SRCFILES}
)

target_link_libraries(picongpu  ${LIBS} ${CUDA_CUDART_LIBRARY} z m)


################################################################################
# Install PIConGPU
################################################################################

install(TARGETS picongpu
         RUNTIME DESTINATION bin)

#file(GLOB scripts_to_copy "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.sh")
#foreach(f ${scripts_to_copy})
#   GET_FILENAME_COMPONENT(file_name ${f} NAME CACHE)                                 
#   install(FILES "${f}" DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_READ
#           OWNER_WRITE GROUP_READ GROUP_EXECUTE)
#endforeach(f)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/" DESTINATION bin
    FILES_MATCHING PATTERN "*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    PATTERN .svn EXCLUDE   
)

# If the installation prefix does not equal extension path, check if folder must be copied.
# If there is no include folder in installation prefix, also copy all missing folders.
if( (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${PIC_EXTENSION_PATH}") OR 
    (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include"))

    #copy all importend subfolders to install folder
    foreach(dir ${PIC_COPY_ON_INSTALL})

      #if source not exists than copy
      if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
          #copy importend subfolders from extension path (default PIConGPU parameter)
          if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
              DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
              PATTERN .svn EXCLUDE
              PATTERN .git EXCLUDE
            )
          endif()

          #copy importend subfolders from extension path (from extension path)
          if(EXISTS "${PIC_EXTENSION_PATH}/${dir}/")
            install(DIRECTORY "${PIC_EXTENSION_PATH}/${dir}/."
               DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
               PATTERN .svn EXCLUDE
               PATTERN .git EXCLUDE
            )
          endif()
      endif()

    endforeach()
endif()
