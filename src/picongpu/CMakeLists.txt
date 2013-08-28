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

cmake_minimum_required(VERSION 2.8)

OPTION(VAMPIR_ENABLE "create picongpu with vampir support" OFF)


IF(VAMPIR_ENABLE)
    message("[CONFIG]  build program with vampir support")
    set(CMAKE_CXX_COMPILER "vtc++")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
    set(CMAKE_CXX_FLAGS_DEFAULT "${CMAKE_CXX_FLAGS_DEFAULT} -vt:hyb -L/$ENV{VT_ROOT}/lib -finstrument-functions-exclude-file-list=${CMAKE_CXX_INST_FILE_FILTER} -finstrument-functions-exclude-function-list=${CMAKE_CXX_INST_FUNC_FILTER} -DVTRACE")
ENDIF(VAMPIR_ENABLE)

# Projekt name
project (PIConGPU)

#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}")

#include cuda
FIND_PACKAGE(CUDA REQUIRED)

OPTION(PIC_RELEASE "disable all runtime asserts" OFF)
IF(PIC_RELEASE)
    SET(CMAKE_BUILD_TYPE Release)
    ADD_DEFINITIONS(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
ELSE(PIC_RELEASE)
   set(CMAKE_BUILD_TYPE Debug)
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g,-pthread")
ENDIF(PIC_RELEASE)

#include MPI
FIND_PACKAGE(MPI REQUIRED)

#include pthreads
FIND_PACKAGE(Threads REQUIRED)
SET(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

#Boost from system
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex)
INCLUDE_DIRECTORIES(AFTER ${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
SET(LIBS ${LIBS} ${Boost_LIBRARIES})

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

########################################################################
#set options

##CUDA##
SET(CUDA_ARCH sm_13 CACHE STRING "set GPU architecture" )
STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_SM13)
SET(CUDA_FTZ "--ftz=false" CACHE STRING "set flush to zero for GPU")
IF( IS_CUDA_ARCH_SM13)
    SET(CUDA_FTZ "")
ENDIF( IS_CUDA_ARCH_SM13)

SET(CUDA_MATH --use_fast_math CACHE STRING "set GPU architecture" )
OPTION(CUDA_SHOW_REGISTER "show kernel register and create PTX" OFF)
OPTION(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps. (folder: nvcc_tmp)" OFF)
OPTION(CUDA_SHOW_CODELINES "show kernel lines in cuda-gdb and cuda-memcheck (need architecture >=sm_20)" OFF)

IF(CUDA_SHOW_CODELINES)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xcompiler -rdynamic -lineinfo)
ENDIF(CUDA_SHOW_CODELINES)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
IF(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v -Xopencc=-LIST:source=on)
ENDIF(CUDA_SHOW_REGISTER)

IF(CUDA_KEEP_FILES)
    MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "nvcc_tmp")
ENDIF(CUDA_KEEP_FILES)

##END CUDA##

##PMacc defines#####

OPTION(PMACC_BLOCKING_KERNEL "activate checks for every kernel call and synch after every kernel call" OFF)
IF(PMACC_BLOCKING_KERNEL)
    ADD_DEFINITIONS(-DPMACC_SYNC_KERNEL=1)
ENDIF(PMACC_BLOCKING_KERNEL)

SET(PMACC_VERBOSE "0" CACHE STRING "set verbose level for libPMacc")
ADD_DEFINITIONS(-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE})

######

##PARAM (overwrite) defines#####

IF(PARAM_OVERWRITES)
    FOREACH(param ${PARAM_OVERWRITES})
       ADD_DEFINITIONS(${param})
    ENDFOREACH(param)
ENDIF(PARAM_OVERWRITES)

######

#######################################################
# load cuda_memtest project
#######################################################
SET(DISBALE_COMPILER_FLAGS_IN_SUBPROJECTS ON)

find_path(CUDA_MEMTEST_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../cuda_memtest"
        DOC "path to cuda_memtest"
        )

add_subdirectory(${CUDA_MEMTEST_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_cuda_memtest")

#######################################################
# load mpiInfo project
#######################################################
find_path(MPI_INFO_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../mpiInfo"
        DOC "path to mpiInfo"
        )

add_subdirectory(${MPI_INFO_DIR} "${CMAKE_CURRENT_BINARY_DIR}/build_mpiInfo")

#######################################################

#####PICONGPU OPTIONS##########

SET(PIC_VERBOSE "1" CACHE STRING "set verbose level for picongpu (default is only physic output)")
ADD_DEFINITIONS(-DPIC_VERBOSE_LVL=${PIC_VERBOSE})

OPTION(PIC_ENABLE_HDF5 "create picongpu with hdf5 support" ON)
IF(PIC_ENABLE_HDF5)
    ADD_DEFINITIONS(-DENABLE_HDF5=1)
    FIND_PACKAGE(HDF5 REQUIRED)
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
    LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})
    SET(LIBS ${LIBS} hdf5)
ENDIF(PIC_ENABLE_HDF5)

OPTION(PIC_ENABLE_PNG "create picongpu png output support (need pngwriter)" ON)
IF(PIC_ENABLE_PNG)
    ADD_DEFINITIONS(-DPIC_ENABLE_PNG=1)
ENDIF(PIC_ENABLE_PNG)

OPTION(PIC_ENABLE_INSITU_VOLVIS "enable In Situ Volume Visualization" OFF)
IF(PIC_ENABLE_INSITU_VOLVIS)
	ADD_DEFINITIONS(-DENABLE_INSITU_VOLVIS=1)
	SET(CUDA_ARCH sm_35 CACHE STRING "set GPU architecture")
	SET(IceT_DIR "$ENV{ICET_ROOT}/lib/")
	FIND_PACKAGE(IceT REQUIRED)
	INCLUDE_DIRECTORIES(${ICET_INCLUDE_DIRS})
	SET(LIBS ${LIBS} ${ICET_CORE_LIBS} ${ICET_MPI_LIBS})
ENDIF(PIC_ENABLE_INSITU_VOLVIS)

#########################################################################
# Warnings
# GNU
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# ICC
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# PGI
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
ENDIF()


#########################################################################
# Configure include directories

#libPMacc

find_path(PMACC_ROOT_DIR
  NAMES include/types.h
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../libPMacc"
  DOC "libPMacc root location."
  )

INCLUDE_DIRECTORIES(${PMACC_ROOT_DIR}/include)

#LIBSPLASH
IF(PIC_ENABLE_HDF5)
  # find splash source code
  find_path(PIC_SPLASH_ROOT_DIR
    NAMES include/sdc_defines.hpp lib/libsplash.so
    PATHS ENV SPLASH_ROOT
    DOC "libsplash ROOT location. (only used if HDF5 is enabled)"
    )

    #add include information
    INCLUDE_DIRECTORIES(${PIC_SPLASH_ROOT_DIR}/include)

    #add library information
    # dynamic / shared
    #LINK_DIRECTORIES("${PIC_SPLASH_ROOT_DIR}/lib")   
    #SET(LIBS ${LIBS} splash)
    # static
    ADD_LIBRARY(splash_static STATIC IMPORTED)
    SET_TARGET_PROPERTIES(splash_static PROPERTIES IMPORTED_LOCATION ${PIC_SPLASH_ROOT_DIR}/lib/libsplash.a)
    SET(LIBS ${LIBS} splash_static)
ENDIF(PIC_ENABLE_HDF5)

#PNGWRITER
IF(PIC_ENABLE_PNG)
  find_path(PIC_PNGWRITER_ROOT_DIR
    NAMES include/pngwriter.h
    PATHS ENV PNGWRITER_ROOT 
    DOC "pngwriter root location. (only used if PIC_ENABLE_PNG is enabled)"
    )
ENDIF(PIC_ENABLE_PNG)
IF(PIC_ENABLE_PNG)
    INCLUDE_DIRECTORIES(${PIC_PNGWRITER_ROOT_DIR}/include /usr/include/freetype2/)
    LINK_DIRECTORIES(${PIC_PNGWRITER_ROOT_DIR}/lib)
    SET(LIBS ${LIBS} png pngwriter freetype)
ENDIF(PIC_ENABLE_PNG)

# check if PIC_EXTENSION_PATH is relativ or absolut

find_path(PIC_EXTENSION_PATH
    NAMES include/simulation_defines/param/componentsConfig.param
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
    DOC "search PATH to extension folder"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )

SET(PIC_COPY_ON_INSTALL "include/simulation_defines" "submit" CACHE LIST "folder which copied on install to install path" )

################################################################################
INCLUDE_DIRECTORIES(include ${MPI_INCLUDE_PATH} ${mpi_include_path})
INCLUDE_DIRECTORIES(BEFORE ${PIC_EXTENSION_PATH}/include)
################################################################################


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${INCLUDE_DIRECTORIES})


IF(VAMPIR_ENABLE)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${CMAKE_CXX_INST_FILE_FILTER}\\\" -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${CMAKE_CXX_INST_FUNC_FILTER}\\\" -Xcompiler=-DVTRACE -Xcompiler=-I/sw/xk6/vampirtrace/5.14.3-iofsl/cle4.1_gnu4.7.2//include/vampirtrace/ -v  )
ENDIF(VAMPIR_ENABLE)


FILE(GLOB CUDASRCFILES "*.cu")
FILE(GLOB SRCFILES "*.cpp")

cuda_add_executable(picongpu
    ${CUDASRCFILES}
    ${SRCFILES}
)

IF(VAMPIR_ENABLE)
    SET(LIBS vt-hyb ${LIBS} )
ENDIF(VAMPIR_ENABLE)

target_link_libraries(picongpu  ${LIBS} ${CUDA_CUDART_LIBRARY} z m ${MPI_LIBRARIES}) # not necessary ${MPI_EXTRA_LIBRARY}


INSTALL(TARGETS picongpu
         RUNTIME DESTINATION bin)

#FILE(GLOB scripts_to_copy "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.sh")
#FOREACH(f ${scripts_to_copy})
#   GET_FILENAME_COMPONENT(file_name ${f} NAME CACHE)                                 
#   INSTALL(FILES "${f}" DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE)
#ENDFOREACH(f)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/" DESTINATION bin
    FILES_MATCHING PATTERN "*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    PATTERN .svn EXCLUDE   
)

#if install prefix is not equal with extension path than check if folder must copyed
# or if we have no include folder in install prefix we also copy all missing folders
IF( (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${PIC_EXTENSION_PATH}") OR (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include"))
    #copy all importend subfolders to install folder
    FOREACH(dir ${PIC_COPY_ON_INSTALL})
      #if source not exists than copy
      IF(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
          #copy importend subfolders from extension path (default picongpu parameter)
          IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/")
            INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
              DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
              PATTERN .svn EXCLUDE
            )
          ENDIF()
          #copy importend subfolders from extension path (from extension path)
          IF(EXISTS "${PIC_EXTENSION_PATH}/${dir}/")
            INSTALL(DIRECTORY "${PIC_EXTENSION_PATH}/${dir}/."
               DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
               PATTERN .svn EXCLUDE
            )
          ENDIF()
      ENDIF()
    ENDFOREACH()
ENDIF()
