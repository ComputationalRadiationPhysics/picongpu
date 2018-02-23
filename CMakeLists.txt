# Copyright 2018 Axel Huebl
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

# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.10.0)

project(PIConGPU VERSION 0.1.0)


# Project structure ###########################################################
#
# temporary build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# install directories
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)
# note: individual apps will install their own CMake config packages as well!
set(CMAKE_INSTALL_CMAKEDIR lib/cmake/PIConGPU)


# Options and Variants ########################################################
#
# default for core-components
option(PIConGPU_INSTALL_DEFAULTS "Select or deselect all components by default" ON)

# control components to install
option(PIConGPU_INSTALL_APP      "Install the PIConGPU application sources"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_PMACC    "Install internally shipped PMacc sources"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_TOOLS    "Install internally shipped tools"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_TBG      "Install internally shipped tbg"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_ALPAKA   "Install internally shipped alpaka"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_CUPLA    "Install internally shipped cupla"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_MALLOCMC "Install internally shipped mallocMC"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_CMAKEM   "Install internally shipped CMake modules"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_MEMTEST  "Install internally shipped CUDA memtest"
       ${PIConGPU_INSTALL_DEFAULTS})
option(PIConGPU_INSTALL_MPIINFO  "Install internally shipped mpiInfo"
       ${PIConGPU_INSTALL_DEFAULTS})
# TODO: BUILD DOCS: doxygen?
# TODO: BUILD DOCS: sphinx?
#option(PIConGPU_INSTALL_DOXYGEN     "Install Doxygen HTML documentation" OFF)
#option(PIConGPU_INSTALL_SPHINX_HTML "Install Sphinx HTML manual" OFF)

# collect all options for status message
set(PIConGPU_CONFIG_OPTIONS APP PMACC TOOLS TBG)

set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the build type, e.g. Release or Debug." FORCE)
endif()


# Dependencies ################################################################
#
if(PIConGPU_INSTALL_APP)
    # add_subdirectory(include/picongpu)
    # TODO: sources: instead of using the CMakeLists.txt do a pure copy?
    # TODO: add cmake config package for the app itself?
endif()
if(PIConGPU_INSTALL_PMACC)
    add_subdirectory(include/pmacc)
    # TODO: cmake config package needs to be generated in include/pmacc/CMakeLists.txt
    # TODO: include/picongpu/CMakeLists.tst: should just do "add_subdirectory"
    #       on PICONGPU_INSTALL_INTERNAL_PMACC and `find_package(PMacc CONFIG REQUIRED)` else
endif()
if(PIConGPU_INSTALL_TOOLS)
    # TODO: build subprojects
    # TODO: rest: pure copy?
endif()
if(PIConGPU_INSTALL_TBG)
    # pure copy?
endif()
if(PIConGPU_INSTALL_DOXYGEN)
    # TODO: find doxygen
endif()
if(PIConGPU_INSTALL_SPHINX_HTML)
    # TODO: find python dependencies from requirements.txt
endif()

# TODO: do sanity check for full dependencies? Maybe warn only? Maybe as an option?
#       just cover as a "make test" build of an example below?


# Targets #####################################################################
#
# create only targets for the PIConGPUapp


# Generate Files with Configuration Options ###################################
#
# this would be a PIConGPU-framework/superbuild cmake config package -
# but not needed!
#
# TODO: we can still use this to create an "APP" config package!!


# Installs ####################################################################
#
# regular install (copy sources)
# headers, libraries and exectuables
#install(TARGETS PIConGPU EXPORT PIConGPUTargets
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#    RUNTIME DESTINATION bin
#    INCLUDES DESTINATION include
#)
if(PIConGPU_INSTALL_APP)
    install(DIRECTORY "${PIConGPU_SOURCE_DIR}/include/picongpu"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN ".svn" EXCLUDE
        PATTERN ".git" EXCLUDE
    )
endif()

if(PIConGPU_INSTALL_TBG)
    install(FILES
            ${PIConGPU_SOURCE_DIR}/bin/tbg
            ${PIConGPU_SOURCE_DIR}/bin/egetopt
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# CMake package file for find_package(PIConGPU::PIConGPU) in depending projects
#install(EXPORT PIConGPUTargets
#    FILE PIConGPUTargets.cmake
#    NAMESPACE PIConGPU::
#    DESTINATION lib/cmake/PIConGPU
#)
#install(
#    FILES
#        ${PIConGPU_BINARY_DIR}/PIConGPUConfig.cmake
#        ${PIConGPU_BINARY_DIR}/PIConGPUConfigVersion.cmake
#    DESTINATION lib/cmake/PIConGPU
#)

# TODO: development install (create symlinks)
# https://github.com/ComputationalRadiationPhysics/picongpu/issues/1969


# Tests #######################################################################
#
# enable_testing()
# TODO: test-build a PIConGPU example on installed targets?


# Status Message for Build Options ############################################
#
message("")
message("PIConGPU framework build configuration:")
if(PIConGPU_INSTALL_APP)
    message("  PIConGPU Version: ${PIConGPUapp_VERSION}")
endif()
if(PIConGPU_INSTALL_PMACC)
    message("  PMacc Version: ${PMacc_VERSION}")
endiF()
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("        bin: ${CMAKE_INSTALL_BINDIR}")
message("        lib: ${CMAKE_INSTALL_LIBDIR}")
message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("      cmake: ${CMAKE_INSTALL_CMAKEDIR}")
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
message("  Build Options:")

foreach(opt IN LISTS PIConGPU_CONFIG_OPTIONS)
    if(${PIConGPU_INSTALL_${opt}})
        message("    ${opt}: ON")
    else()
        message("    ${opt}: OFF")
    endif()
endforeach()
message("")
