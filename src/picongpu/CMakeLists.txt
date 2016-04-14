# Copyright 2013-2016 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau,
#                     Rene Widera, Alexander Grund
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

cmake_minimum_required(VERSION 3.1.0)


################################################################################
# Project 
################################################################################

project(PIConGPU)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HDF5_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")
# Last add generic system path to the end (as last fallback)
list(APPEND "/usr/lib/x86_64-linux-gnu/")

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/cmake-modules/)


################################################################################
# C++11 Enabler
################################################################################

# By using this if-else one can assume that:
# CMAKE_CXX_STANDARD==11 <=> CUDA_NVCC_FLAGS contain "-std=c++11"
if("${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11")
    set(CMAKE_CXX_STANDARD 11)
elseif("${CMAKE_CXX_STANDARD}" STREQUAL "11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
else()
    # Note: Add support for C++14 once CUDA supports it (maybe 8.0?)
    set(CMAKE_CXX_STANDARD 98)
endif()

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

if( (CUDA_VERSION VERSION_GREATER 7.0) AND
    (CMAKE_CXX_STANDARD EQUAL 98) )
    # Kind of invalid usage of floats in constant expressions
    # which causes problems in nvcc 7.5 and up:
    # https://github.com/ComputationalRadiationPhysics/picongpu/issues/1290
    message(FATAL_ERROR "Please enable C++11 with
                        `-std=c++11` or `-DCMAKE_CXX_STANDARD=11`
                        when compiling with nvcc 7.5")
endif()

################################################################################
# VampirTrace
################################################################################

option(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)

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
    add_definitions(-DVTRACE)

    # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
    add_definitions(-DMPICH_IGNORE_CXX_SEEK)
endif(VAMPIR_ENABLE)


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

find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options regex filesystem
                                              system thread math_tr1)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# Boost 1.55 added support for a define that makes result_of look for
# the result<> template and falls back to decltype if none is found. This is
# great for the transition from the "wrong" usage to the "correct" one as
# both can be used. But:
# 1) Cannot be used in 7.0 due to nvcc bug:
#    http://stackoverflow.com/questions/31940457/
# 2) Requires C++11 enabled as there is no further check in boost besides
#    the version check of nvcc
if( (NOT CUDA_VERSION VERSION_EQUAL 7.0) AND (CMAKE_CXX_STANDARD EQUAL 11) )
    add_definitions(-DBOOST_RESULT_OF_USE_TR1_WITH_DECLTYPE_FALLBACK)
else()
    # Fallback
    add_definitions(-DBOOST_RESULT_OF_USE_TR1)
endif()

# work-arounds and known issues
if( (Boost_VERSION EQUAL 106000) AND
    (CMAKE_CXX_STANDARD EQUAL 98) )
    # Boost Bug https://svn.boost.org/trac/boost/ticket/11852
    message(FATAL_ERROR "Boost: Please use a C++11 enabled Compiler with "
                        "`-std=c++11` when compiling with Boost 1.60.0")
endif()

# Boost 1.60.0 and CUDA releases prior to 7.5 fail on variadic templates
# when used with C++11
if( (Boost_VERSION EQUAL 106000) AND
    (CUDA_VERSION VERSION_LESS 7.5) AND
    (NOT CMAKE_CXX_STANDARD EQUAL 98) )
    # Boost Bug https://svn.boost.org/trac/boost/ticket/11897
    message(STATUS "Boost: Disable variadic templates")
    add_definitions(-DBOOST_NO_CXX11_VARIADIC_TEMPLATES)
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
  set(MALLOCMC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/mallocMC")
  find_package(mallocMC 2.2.0 REQUIRED)
endif(NOT mallocMC_FOUND)

include_directories(SYSTEM ${mallocMC_INCLUDE_DIRS})
add_definitions(${mallocMC_DEFINITIONS})
set(LIBS ${LIBS} ${mallocMC_LIBRARIES})


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
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdParty/cuda_memtest"
        DOC "path to cuda_memtest"
        )

add_subdirectory(${CUDA_MEMTEST_DIR}
                 "${CMAKE_CURRENT_BINARY_DIR}/build_cuda_memtest")


################################################################################
# Load mpiInfo project
################################################################################

find_path(MPI_INFO_DIR
        NAMES CMakeLists.txt
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../mpiInfo"
        DOC "path to mpiInfo"
        )

add_subdirectory(${MPI_INFO_DIR}
                 "${CMAKE_CURRENT_BINARY_DIR}/build_mpiInfo")


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
# ADIOS
################################################################################

# find adios installation
#   set(ADIOS_USE_STATIC_LIBS ON) # force static linking
find_package(ADIOS 1.9.0)

if(ADIOS_FOUND)
    add_definitions(-DENABLE_ADIOS=1)
    include_directories(SYSTEM ${ADIOS_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ADIOS_LIBRARIES})
endif(ADIOS_FOUND)


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
# Configure include directories
################################################################################

# libPMacc

set(PMACC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libPMacc")
include_directories(${PMACC_ROOT_DIR}/include)


################################################################################
# libSplash (+ hdf5 due to required headers)
################################################################################

# find libSplash installation
# prefer static libraries over shared ones (but do not force them)
set(Splash_USE_STATIC_LIBS ON)
find_package(Splash 1.4.0 COMPONENTS PARALLEL)

if(Splash_FOUND)
    include_directories(SYSTEM ${Splash_INCLUDE_DIRS})
    list(APPEND Splash_DEFINITIONS "-DENABLE_HDF5=1")
    add_definitions(${Splash_DEFINITIONS})
    set(LIBS ${LIBS} ${Splash_LIBRARIES})
endif(Splash_FOUND)


################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.5.6)

if(PNGwriter_FOUND)
    include_directories(SYSTEM ${PNGwriter_INCLUDE_DIRS})
    list(APPEND PNGwriter_DEFINITIONS "-DPIC_ENABLE_PNG=1")
    add_definitions(${PNGwriter_DEFINITIONS})
    set(LIBS ${LIBS} ${PNGwriter_LIBRARIES})
endif(PNGwriter_FOUND)


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

    #copy all important subfolders to install folder
    foreach(dir ${PIC_COPY_ON_INSTALL})

      #if source not exists than copy
      if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
          #copy important subfolders from extension path (default PIConGPU parameter)
          if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/")
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/."
              DESTINATION "${CMAKE_INSTALL_PREFIX}/${dir}"
              PATTERN .svn EXCLUDE
              PATTERN .git EXCLUDE
            )
          endif()

          #copy important subfolders from extension path (from extension path)
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
