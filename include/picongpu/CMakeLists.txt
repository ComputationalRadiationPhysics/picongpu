# Copyright 2013-2022 Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau,
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

cmake_minimum_required(VERSION 3.18.0)


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
list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENPMD_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

###############################################################################
# Misc
###############################################################################

# Create options to configure the usage of a dependency
#
# The status of the dependency must be set by calling 'PIC_dependency_set_status'.
#
# Creates a variable named 'PIC_SEARCH_<name>' which is true if 'PIC_USE_<name>' is AUTO or ON.
# Creates a list 'PIC_DEPENDENCY_LIST' with all dependencies to create a status report at the end of the CMake script.
# Creates a variable 'PIC_HAVE_<name>' with the status TRUE means dependency is found else FALSE/empty.
#
#   name        - dependency name
#   description - description for the created option named 'PIC_USE_<name>'
#   default     - default value of the option: ON,OFF, or AUTO
function(PIC_dependency name description default)
    set(PIC_DEPENDENCY_LIST ${PIC_DEPENDENCY_LIST} ${name} PARENT_SCOPE)

    set(PIC_USE_${name} ${default} CACHE STRING "${description}")
    set_property(CACHE PIC_USE_${name} PROPERTY STRINGS "AUTO;ON;OFF")

    if(PIC_USE_${name} STREQUAL AUTO OR PIC_USE_${name} STREQUAL ON)
        set(PIC_SEARCH_${name} TRUE PARENT_SCOPE)
    else()
        set(PIC_SEARCH_${name} FALSE PARENT_SCOPE)
    endif()
    set(PIC_HAVE_${name} PARENT_SCOPE)
endfunction()

# Set status of a dependency
#
# If function is called twice the second call will have no effect.
# Set the variable 'PIC_HAVE_<name>' depending on the optional parameter '...'.
#
#   name - dependency name
#   ...  - TRUE/FALSE or content of a variable (can be empty)
function(PIC_dependency_set_status name)
    # handle optional arguments to support passing not existing <project>_FOUND variables
    if(ARGC EQUAL 2)
        set(value ${ARGN})
    else()
        set(value FALSE)
    endif()
    if(${value} AND NOT PIC_HAVE_${name}_LOCKED AND PIC_SEARCH_${name})
        set(PIC_HAVE_${name} TRUE PARENT_SCOPE)
    else()
        set(PIC_HAVE_${name} FALSE PARENT_SCOPE)
        if(PIC_USE_${name} STREQUAL ON)
            message(FATAL_ERROR "Optional dependency '${name}' not found but explicitly requested with 'PIC_USE_${name}=ON'.")
        endif()
    endif()
    # guard to set the value only once
    set(PIC_HAVE_${name}_LOCKED TRUE PARENT_SCOPE)
endfunction()

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

# enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)


################################################################################
# Disallow in-source build
################################################################################

get_filename_component(SOURCE_DIR_ROOT ${PIConGPUapp_SOURCE_DIR}/../.. ABSOLUTE)
string(FIND "${PIConGPUapp_BINARY_DIR}"
            "${SOURCE_DIR_ROOT}/" IN_SRC_POS)
#if(IN_SRC_POS GREATER -1)
#  message(FATAL_ERROR
#    "PIConGPU requires an out of source build. "
#    "Please remove \n"
#    "  - CMakeCache.txt\n"
#    "  - CMakeFiles/\n"
#    "and create a separate build directory. "
#    "See: INSTALL.rst")
#endif()

unset(IN_SRC_POS)


################################################################################
# Find PMacc
################################################################################

find_package(PMacc REQUIRED CONFIG PATHS "${PIConGPUapp_SOURCE_DIR}/../pmacc")

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
set(HOST_LIBS ${HOST_LIBS} ${MPI_C_LIBRARIES})


################################################################################
# Find PThreads
################################################################################

find_package(Threads REQUIRED)
set(HOST_LIBS ${HOST_LIBS} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# Find math from stdlib
################################################################################

if(NOT WIN32 AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # automatically added on windows
    # should not be added for Intel compiler as it has an optimized lib included
    set(HOST_LIBS ${HOST_LIBS} m)
endif()


################################################################################
# Find Boost
################################################################################

find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options)
if(TARGET Boost::boost)
    set(HOST_LIBS ${HOST_LIBS} Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    set(HOST_LIBS ${HOST_LIBS} ${Boost_LIBRARIES})
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

if( NOT (alpaka_ACC_GPU_CUDA_ENABLE OR alpaka_ACC_GPU_HIP_ENABLE) OR alpaka_CUDA_COMPILER MATCHES "clang")
    # cuda_memtest is not compiling with host accelerator of alpaka or clang
    set(CUDAMEMTEST_ENABLE OFF CACHE BOOL "Build cuda_memtest and the helper mpiInfo \
                           (allow GPU health test before running PIConGPU)" FORCE)
    message(STATUS "Disable 'cuda_memtest' build: not supported by clang or non CUDA/HIP alpaka backends")
endif()

if(CUDAMEMTEST_ENABLE)
    if(alpaka_ACC_GPU_CUDA_ENABLE)
        # Enabling CUDA solves a bug in mpiInfo
        # Error message: No known features for CUDA compiler "" version .
        ENABLE_LANGUAGE(CUDA)
    endif()
    set(SAME_NVCC_FLAGS_IN_SUBPROJECTS OFF)
    find_path(CUDA_MEMTEST_DIR
            NAMES cuda_memtest.h
            PATHS "${PIConGPUapp_SOURCE_DIR}/../../thirdParty/cuda_memtest"
            DOC "path to cuda_memtest"
    )
    if(alpaka_ACC_GPU_HIP_ENABLE)
        set(CUDA_MEMTEST_BACKEND hip CACHE STRING "Select the backend API used for the test." FORCE)
    endif()
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

option(PIC_ADD_RPATH "Add RPATH's to binaries." ON)


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

option(PIC_CI_COMPILE "Enable strict error checking." OFF)
if(PIC_CI_COMPILE)
    # GNU
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
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
endif()


################################################################################
# openPMD
################################################################################

PIC_dependency(openPMD "API for IO" AUTO)

if(PIC_SEARCH_openPMD)
    # find openPMD installation
    find_package(openPMD 0.14.3 CONFIG COMPONENTS MPI)
    if(openPMD_FOUND)
        if(openPMD_HAVE_ADIOS2 OR openPMD_HAVE_HDF5)
            message(STATUS "Found openPMD: ${openPMD_DIR}")
            add_definitions(-DENABLE_OPENPMD=1)

            # none of these should appear in cmake-gui, so make them internal
            set(JSON_BuildTests OFF CACHE INTERNAL "")
            set(JSON_MultipleHeaders OFF CACHE INTERNAL "")
            set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
            set(JSON_Install OFF CACHE INTERNAL "")  # only used PRIVATE

            # allow using externally installed nlohmann_json
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

            # TOML
            set(
                PIC_toml11_PROVIDER "intern" CACHE
                STRING "Use internally shipped or external nlohmann_json library.")
            set_property(
                CACHE PIC_toml11_PROVIDER
                PROPERTY STRINGS "intern;extern")
            mark_as_advanced(PIC_toml11_PROVIDER)
            if(${PIC_toml11_PROVIDER} STREQUAL "intern")
                add_subdirectory(
                    "${PIConGPUapp_SOURCE_DIR}/../../thirdParty/toml11"
                    "${CMAKE_CURRENT_BINARY_DIR}/build_toml11"
                    # EXCLUDE_FROM_ALL ensures that toml11 is not part of
                    # make install.
                    # The library is header-only and linking it against
                    # PIConGPU targets is sufficient.
                    # It needs not be part of any build or install targets
                    # explicitly.
                    EXCLUDE_FROM_ALL)
            else()
                find_package(toml11 3.7.0 CONFIG REQUIRED)
                message(STATUS "toml11: Found version '${toml11_VERSION}'")
            endif()
            set(HOST_LIBS ${HOST_LIBS} openPMD::openPMD)
        else()
            message(STATUS "Found openPMD at ${openPMD_DIR}, but PIConGPU requires"
                           " availability of either its ADIOS2 or HDF5 backend - "
                           "NOT BUILDING the openPMD plugin")
            PIC_dependency_set_status(openPMD FALSE)
        endif()
    else(openPMD_FOUND)
        message(STATUS "Could NOT find openPMD - set openPMD_DIR or check your CMAKE_PREFIX_PATH")
    endif(openPMD_FOUND)
endif()

PIC_dependency_set_status(openPMD ${openPMD_FOUND})

################################################################################
# PNGwriter
################################################################################

PIC_dependency(PNGwriter "API for IO with HDF5" AUTO)

if(PIC_SEARCH_PNGwriter)
    # find PNGwriter installation
    find_package(PNGwriter 0.7.0 CONFIG)

    if(PNGwriter_FOUND)
        set(HOST_LIBS ${HOST_LIBS} PNGwriter::PNGwriter)
        add_definitions(-DPIC_ENABLE_PNG=1)
        message(STATUS "Found PNGwriter: ${PNGwriter_DIR}")
    else()
        message(STATUS "Could NOT find PNGwriter - "
                       "set PNGwriter_DIR or check your CMAKE_PREFIX_PATH")
        PIC_dependency_set_status(PNGwriter FALSE)
    endif(PNGwriter_FOUND)
endif()

PIC_dependency_set_status(PNGwriter ${PNGwriter_FOUND})

################################################################################
# ISAAC
################################################################################

PIC_dependency(ISAAC "Framework to in-situ visualize PIConGPU during the execution." AUTO)

if(PIC_SEARCH_ISAAC)
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
        PIC_dependency_set_status(ISAAC FALSE)
    endif(ISAAC_FOUND)
endif()

PIC_dependency_set_status(ISAAC ${ISAAC_FOUND})

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
target_link_libraries(picongpu-hostonly PUBLIC ${HOST_LIBS})
# Some host compiled libraries depend on std::filesystem,
# compatibility to std::experimental::filesystem is handled in pmacc.
target_link_libraries(picongpu-hostonly PRIVATE pmacc::filesystem)

cupla_add_executable(picongpu
     ${ACCSRCFILES}
)

target_link_libraries(picongpu PUBLIC ${LIBS} ${HOST_LIBS} picongpu-hostonly)
target_link_libraries(picongpu PRIVATE pmacc::pmacc)

if(PIC_HAVE_openPMD)
    # Including <nlohmann/json.hpp> will throw loads of warnings. Quiet them.
    # (Doesn't work for nvcc??)
    target_include_directories(
        picongpu-hostonly
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(picongpu-hostonly PRIVATE nlohmann_json::nlohmann_json)

    # same tricks for toml11
    target_include_directories(
        picongpu-hostonly
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:toml11::toml11,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(picongpu-hostonly PRIVATE toml11::toml11)
endif()

## annotate with RPATH's
if(PIC_ADD_RPATH)
    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        if(APPLE)
            set_target_properties(picongpu PROPERTIES INSTALL_RPATH "@loader_path")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
            set_target_properties(picongpu PROPERTIES INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
    if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        set_target_properties(picongpu PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
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
            -std=c++17
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

message("")
message("Optional Dependencies:")
foreach(dep IN LISTS PIC_DEPENDENCY_LIST)
    if(PIC_HAVE_${dep})
        message("  ${dep}: ON")
    else()
        message("  ${dep}: OFF")
    endif()
endforeach()
message("")

