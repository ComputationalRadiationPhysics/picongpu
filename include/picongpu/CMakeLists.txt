# Copyright 2013-2021 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau,
#                     Rene Widera, Alexander Grund, Alexander Matthes,
#                     Franz Poeschel, Richard Pausch
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

cmake_minimum_required(VERSION 3.15.0)


################################################################################
# Project
################################################################################

project(PIConGPUapp)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HDF5_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{ADIOS_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENPMD_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# Language Flags
###############################################################################

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)


################################################################################
# Disallow in-source build
################################################################################

get_filename_component(SOURCE_DIR_ROOT ${PIConGPUapp_SOURCE_DIR}/../.. ABSOLUTE)
string(FIND "${PIConGPUapp_BINARY_DIR}"
            "${SOURCE_DIR_ROOT}/" IN_SRC_POS)
if(IN_SRC_POS GREATER -1)
  message(FATAL_ERROR
    "PIConGPU requires an out of source build. "
    "Please remove \n"
    "  - CMakeCache.txt\n"
    "  - CMakeFiles/\n"
    "and create a separate build directory. "
    "See: INSTALL.rst")
endif()

unset(IN_SRC_POS)


################################################################################
# Find PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${PIConGPUapp_SOURCE_DIR}/../pmacc")
include_directories(SYSTEM ${PMacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${PMacc_LIBRARIES})
add_definitions(${PMacc_DEFINITIONS})

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(LIBS ${LIBS} ${MPI_C_LIBRARIES})


################################################################################
# Find PThreads
################################################################################

find_package(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# Find OpenMP
################################################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ALPAKA_ACC_GPU_CUDA_ENABLE AND ALPAKA_CUDA_COMPILER MATCHES "clang")
    message(WARNING "OpenMP host side acceleration is disabled: CUDA compilation with clang is not supporting OpenMP.")
else()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


################################################################################
# Find zlib
################################################################################

find_package(ZLIB REQUIRED)
set(LIBS ${LIBS} ZLIB::ZLIB)


################################################################################
# Find math from stdlib
################################################################################

if(NOT WIN32 AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # automatically added on windows
    # should not be added for Intel compiler as it has an optimized lib included
    set(LIBS ${LIBS} m)
endif()


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.65.1 REQUIRED COMPONENTS program_options filesystem
                                              system math_tr1 serialization)
if(TARGET Boost::program_options)
    set(LIBS ${LIBS} Boost::boost Boost::program_options
                     Boost::filesystem Boost::system Boost::math_tr1
                     Boost::serialization)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${Boost_LIBRARIES})
endif()


################################################################################
# PARAM (overwrite) defines
################################################################################

if(PARAM_OVERWRITES)
    foreach(param ${PARAM_OVERWRITES})
       add_definitions(${param})
    endforeach(param)
endif(PARAM_OVERWRITES)


################################################################################
# load cuda_memtest and mpiInfo projects
################################################################################

option(CUDAMEMTEST_ENABLE "Build cuda_memtest and the helper mpiInfo \
                          (allow GPU health test before running PIConGPU)" ON)

if((NOT ALPAKA_ACC_GPU_CUDA_ENABLE) OR ALPAKA_CUDA_COMPILER MATCHES "clang")
    # cuda_memtest is not compiling with host accelerator of alpaka or clang
    set(CUDAMEMTEST_ENABLE OFF CACHE BOOL "Build cuda_memtest and the helper mpiInfo \
                           (allow GPU health test before running PIConGPU)" FORCE)
    message(STATUS "Disable 'cuda_memtest' build: not supported by clang or non CUDA alpaka backends")
endif()

if(CUDAMEMTEST_ENABLE)
    set(SAME_NVCC_FLAGS_IN_SUBPROJECTS OFF)
    find_path(CUDA_MEMTEST_DIR
            NAMES cuda_memtest.h
            PATHS "${PIConGPUapp_SOURCE_DIR}/../../thirdParty/cuda_memtest"
            DOC "path to cuda_memtest"
    )
    add_subdirectory(${CUDA_MEMTEST_DIR}
                     "${CMAKE_CURRENT_BINARY_DIR}/build_cuda_memtest")


    # mpiInfo utility
    find_path(MPI_INFO_DIR
            NAMES mpiInfo.cpp
            PATHS "${PIConGPUapp_SOURCE_DIR}/../mpiInfo"
            DOC "path to mpiInfo"
    )
    add_subdirectory(${MPI_INFO_DIR}
                     "${CMAKE_CURRENT_BINARY_DIR}/build_mpiInfo")
endif()


################################################################################
# PIConGPU options
################################################################################

set(PIC_VERBOSE "1" CACHE STRING
    "Set verbosity level for PIConGPU (default is only physics output)")
add_definitions(-DPIC_VERBOSE_LVL=${PIC_VERBOSE})


################################################################################
# Additional defines for PIConGPU outputs
################################################################################

# CMake
add_definitions(-DCMAKE_VERSION=${CMAKE_VERSION})
add_definitions(-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

# OS & Host Architecture
add_definitions(-DCMAKE_SYSTEM=${CMAKE_SYSTEM})
add_definitions(-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})

# Compiler
add_definitions(-DCMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID})
add_definitions(-DCMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION})


################################################################################
# Warnings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
# ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
# MSVS
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    # Required for MSVS 15.8+ due to alignment change
    # https://devblogs.microsoft.com/cppblog/stl-features-and-fixes-in-vs-2017-15-8/
    if(MSVC_VERSION GREATER_EQUAL 1916)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D _ENABLE_EXTENDED_ALIGNED_STORAGE")
    endif()
endif()


################################################################################
# openPMD
################################################################################

# find openPMD installation
find_package(openPMD 0.12.0 CONFIG COMPONENTS MPI)

if(openPMD_FOUND)
    if(openPMD_HAVE_ADIOS2 OR openPMD_HAVE_HDF5)
        message(STATUS "Found openPMD: ${openPMD_DIR}")
        add_definitions(-DENABLE_OPENPMD=1)

        # non of these should appear in cmake-gui, so make them internal
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_MultipleHeaders OFF CACHE INTERNAL "")
        set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
        set(JSON_Install OFF CACHE INTERNAL "")  # only used PRIVATE

        # allow to use externally installed nlohmann_json
        set(
            PIC_nlohmann_json_PROVIDER "intern" CACHE
            STRING "Use internally shipped or external nlohmann_json library.")
        set_property(
            CACHE PIC_nlohmann_json_PROVIDER
            PROPERTY STRINGS "intern;extern")
        mark_as_advanced(PIC_nlohmann_json_PROVIDER)
        if(${PIC_nlohmann_json_PROVIDER} STREQUAL "intern")
            add_subdirectory(
                "${PIConGPUapp_SOURCE_DIR}/../../thirdParty/nlohmann_json"
                "${CMAKE_CURRENT_BINARY_DIR}/build_nlohmann_json")
        else()
            find_package(nlohmann_json 3.9.1 CONFIG REQUIRED)
            message(STATUS "nlohmann-json: Found version '${nlohmann_json_VERSION}'")
        endif()
        set(LIBS ${LIBS} openPMD::openPMD)
    else()
        message(STATUS "Found openPMD at ${openPMD_DIR}, but PIConGPU requires"
                       " availability of either its ADIOS2 or HDF5 backend - "
                       "NOT BUILDING the openPMD plugin")
    endif()
else(openPMD_FOUND)
    message(STATUS "Could NOT find openPMD - set openPMD_DIR or check your CMAKE_PREFIX_PATH")
endif(openPMD_FOUND)


################################################################################
# ADIOS
################################################################################

# find adios installation
#   set(ADIOS_USE_STATIC_LIBS ON) # force static linking
find_package(ADIOS 1.13.1)

if(ADIOS_FOUND)
    add_definitions(-DENABLE_ADIOS=1)
    include_directories(SYSTEM ${ADIOS_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ADIOS_LIBRARIES})
endif(ADIOS_FOUND)


################################################################################
# libSplash (+ hdf5 due to required headers)
################################################################################

find_package(Splash 1.7.0 CONFIG COMPONENTS PARALLEL)

if(TARGET Splash::Splash)
    add_definitions(-DENABLE_HDF5=1)
    set(LIBS ${LIBS} Splash::Splash)
    message(STATUS "Found Splash: ${Splash_DIR}")
else()
    message(STATUS "Could NOT find Splash - "
                   "set Splash_DIR or check your CMAKE_PREFIX_PATH")
endif()


################################################################################
# PNGwriter
################################################################################

# find PNGwriter installation
find_package(PNGwriter 0.7.0 CONFIG)

if(PNGwriter_FOUND)
    set(LIBS ${LIBS} PNGwriter::PNGwriter)
    add_definitions(-DPIC_ENABLE_PNG=1)
    message(STATUS "Found PNGwriter: ${PNGwriter_DIR}")
else()
    message(STATUS "Could NOT find PNGwriter - "
                   "set PNGwriter_DIR or check your CMAKE_PREFIX_PATH")
endif(PNGwriter_FOUND)


################################################################################
# ISAAC
################################################################################

SET(ISAAC_CUDA OFF CACHE BOOL "Using CUDA")
SET(ISAAC_ALPAKA ON CACHE BOOL "Using Alpaka")
find_package(ISAAC 1.4.0 CONFIG QUIET)
if(ISAAC_FOUND)
    message(STATUS "Found ISAAC: ${ISAAC_DIR}")
    SET(ISAAC_STEREO "No" CACHE STRING "Using stereoscopy")
    SET_PROPERTY(CACHE ISAAC_STEREO PROPERTY STRINGS No SideBySide Anaglyph)

    if(${ISAAC_STEREO} STREQUAL "No")
        add_definitions(-DISAAC_STEREO=0)
    endif()
    if(${ISAAC_STEREO} STREQUAL "SideBySide")
        add_definitions(-DISAAC_STEREO=1)
    endif()
    if(${ISAAC_STEREO} STREQUAL "Anaglyph")
        add_definitions(-DISAAC_STEREO=2)
    endif()

    include_directories(SYSTEM ${ISAAC_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${ISAAC_LIBRARIES})

    set(ISAAC_MAX_FUNCTORS "3" CACHE STRING "Max length of the isaac functor chain" )
    set(ISAAC_DEFAULT_WEIGHT "7" CACHE STRING "Default weight of an isaac source" )
    add_definitions(${ISAAC_DEFINITIONS})
    add_definitions(-DISAAC_MAX_FUNCTORS=${ISAAC_MAX_FUNCTORS})
    add_definitions(-DISAAC_FUNCTOR_POW_ENABLED=0)
    add_definitions(-DISAAC_DEFAULT_WEIGHT=${ISAAC_DEFAULT_WEIGHT})

    add_definitions(-DENABLE_ISAAC=1)
else(ISAAC_FOUND)
    if(DEFINED ISAAC_DEPENDENCY_HINTS)
        message(STATUS "ISAAC was found, but detected the following "
                       "problems:" ${ISAAC_DEPENDENCY_HINTS})
    else()
        message(STATUS "Could NOT find ISAAC - set ISAAC_DIR or check your CMAKE_PREFIX_PATH")
    endif()
endif(ISAAC_FOUND)

################################################################################
# PIConGPU Workarounds
################################################################################

set(PIC_COMPUTE_CURRENT_THREAD_LIMITER_DEFAULT OFF)
if(ALPAKA_ACC_GPU_HIP_ENABLE)
    set(PIC_COMPUTE_CURRENT_THREAD_LIMITER_DEFAULT ON)
endif()
option(PIC_COMPUTE_CURRENT_THREAD_LIMITER "Compute current results with HIP alpaka backend are wrong when more threads than particles in a frame will be used (possible compiler BUG).\
 ON means the number of threads will be limited to number of particles in a frame." ${PIC_COMPUTE_CURRENT_THREAD_LIMITER_DEFAULT})

if(PIC_COMPUTE_CURRENT_THREAD_LIMITER)
    add_definitions(-DPIC_COMPUTE_CURRENT_THREAD_LIMITER=1)
endif()

################################################################################
# Check if PIC_EXTENSION_PATH is relative or absolute
################################################################################

find_path(PIC_EXTENSION_PATH
    NAMES include/picongpu/param/components.param
    PATHS "${PIConGPUapp_SOURCE_DIR}"
    DOC "search PATH to extension folder"
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    )

set(PIC_COPY_ON_INSTALL "include/picongpu;etc/picongpu;lib"
    CACHE STRING
    "Folders that are copied to installation path during install" )

# enforce that all picongpu includes must be prefixed with `picongpu/`
include_directories(..)
include_directories(BEFORE ${PIC_EXTENSION_PATH}/include)


################################################################################
# Compile & Link PIConGPU
################################################################################

# files with accelerator code (kernel and API calls)
list(APPEND ACCSRCFILES
    "${PIConGPUapp_SOURCE_DIR}/main.cpp"
    "${PIConGPUapp_SOURCE_DIR}/versionFormat.cpp"
)

# host-only sources
file(GLOB_RECURSE SRCFILES "*.cpp")
list(REMOVE_ITEM SRCFILES ${ACCSRCFILES})

add_library(picongpu-hostonly
    STATIC
    ${SRCFILES}
)
target_link_libraries(picongpu-hostonly PUBLIC ${LIBS})

cupla_add_executable(picongpu
     ${ACCSRCFILES}
)

target_link_libraries(picongpu PUBLIC ${LIBS} picongpu-hostonly)

if(openPMD_FOUND)
    # Including <nlohmann/json.hpp> will throw loads of warnings. Quiet them.
    # (Doesn't work for nvcc??)
    target_include_directories(
        picongpu-hostonly
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(picongpu-hostonly PRIVATE nlohmann_json::nlohmann_json)
endif()


################################################################################
# Clang-Tidy (3.9+) Target for CI
################################################################################

find_program(
    CLANG_TIDY_BIN
    clang-tidy
    NAMES clang-tidy clang-tidy-3.9 clang-tidy-4.0
    DOC "Path to clang-tidy in version 3.9 or newer"
)
execute_process(
    COMMAND ${CLANG_TIDY_BIN} --version
    RESULT_VARIABLE CLANG_TIDY_RETURN
    OUTPUT_VARIABLE CLANG_TIDY_OUTPUT
)
if(${CLANG_TIDY_RETURN} EQUAL 0)
    string(REGEX MATCH "[0-9]+(\\.[0-9]+)+"
           CLANG_TIDY_VERSION ${CLANG_TIDY_OUTPUT})

    if("${CLANG_TIDY_VERSION}" VERSION_GREATER_EQUAL "3.9.0")
        get_directory_property(ALL_INCLUDES DIRECTORY ${PIConGPUapp_SOURCE_DIR} INCLUDE_DIRECTORIES)
        get_directory_property(ALL_DEFINES DIRECTORY ${PIConGPUapp_SOURCE_DIR} COMPILE_DEFINITIONS)
        foreach(i ${ALL_INCLUDES})
            list(APPEND ALL_INCLUDES_STR "-I${i}")
        endforeach()
        foreach(d ${ALL_DEFINES})
            list(APPEND ALL_DEFINES_STR "-D${d}")
        endforeach()

        add_custom_target(
            clang-tidy
            COMMAND ${CLANG_TIDY_BIN}
            ${SRCFILES} ${ACCSRCFILES}
            -header-filter='.*'
            -config=''
            # -warnings-as-errors='*'
            # -checks='-*,modernize-use-using'
            # -fix # -fix-errors
            --
            -std=c++14
            ${OpenMP_CXX_FLAGS}
            ${ALL_INCLUDES_STR}
            ${ALL_DEFINES_STR}
        )
    endif()
endif()


################################################################################
# Install PIConGPU
################################################################################

install(TARGETS picongpu
         RUNTIME DESTINATION bin)

#file(GLOB scripts_to_copy "${PIConGPUapp_SOURCE_DIR}/bin/*.sh")
#foreach(f ${scripts_to_copy})
#   GET_FILENAME_COMPONENT(file_name ${f} NAME CACHE)
#   install(FILES "${f}" DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_READ
#           OWNER_WRITE GROUP_READ GROUP_EXECUTE)
#endforeach(f)

install(DIRECTORY "${PIConGPUapp_SOURCE_DIR}/../../bin/" DESTINATION bin
    FILES_MATCHING PATTERN "*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    PATTERN .svn EXCLUDE
)

# If the installation prefix does not equal extension path, check if folder must be copied.
# If there is no include folder in installation prefix, also copy all missing folders.
if( (NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${PIC_EXTENSION_PATH}") OR
    (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include"))

    # copy all important subfolders to install folder
    foreach(dir ${PIC_COPY_ON_INSTALL})

      #if source not exists than copy
      if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/${dir}")
          #copy important subfolders from extension path (default PIConGPU parameter)
          if(EXISTS "${PIConGPUapp_SOURCE_DIR}/../../${dir}/")
            install(DIRECTORY "${PIConGPUapp_SOURCE_DIR}/../../${dir}/."
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
