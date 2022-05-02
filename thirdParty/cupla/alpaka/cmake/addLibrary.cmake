#
# Copyright 2015-2021 Benjamin Worpitz, Maximilian Knespel, Simeon Ehrig
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

#------------------------------------------------------------------------------
#
# alpaka_ADD_LIBRARY( cuda_target file0 file1 ... [STATIC | SHARED | MODULE]
#   [EXCLUDE_FROM_ALL] [OPTIONS <nvcc-flags> ... ] )
#
# The position of STATIC, SHARED, MODULE, EXCLUDE_FROM_ALL options don't matter.
# This also means you won't be able to include files with those exact same
# case-sensitive names.
# After OPTIONS only nvcc compiler flags are allowed though. And for readiblity
# and portability you shouldn't completely mix STATIC, ... with the source
# code filenames!
# OPTIONS and the arguments thereafter are ignored if not using CUDA, they
# won't throw an error in that case.
MACRO(alpaka_ADD_LIBRARY libraryName)
    # add_library( <name> [STATIC | SHARED | MODULE]
    #              [EXCLUDE_FROM_ALL]
    #              source1 [source2 ...] )

    # traverse arguments and sort them by option and source files
    SET( arguments ${ARGN} )
    SET( optionsEncountered OFF )
    UNSET( libraryType )
    UNSET( excludeFromAll )
    UNSET( optionArguments )
    FOREACH( argument IN LISTS arguments )
        # 1.) check for OPTIONS
        IF( argument STREQUAL "OPTIONS" )
            IF ( optionsEncountered )
                MESSAGE( FATAL_ERROR "[alpaka_ADD_LIBRARY] OPTIONS subcommand specified more than one time. This is not allowed!" )
            ELSE()
                SET( optionsEncountered ON )
            ENDIF()
        ENDIF()

        # 2.) check if inside OPTIONS, because then all other checks are
        # unnecessary although they could give hints about wrong locations
        # of those subcommands
        IF( optionsEncountered )
            LIST( APPEND optionArguments "${argument}" )
            CONTINUE()
        ENDIF()

        # 3.) check for libraryType and EXCLUDE_FROM_ALL
        IF( ( argument STREQUAL "STATIC" ) OR
            ( argument STREQUAL "SHARED" ) OR
            ( argument STREQUAL "MODULE" )
        )
            IF( DEFINED libraryType )
                message( FATAL_ERROR "Setting more than one library type option ( STATIC SHARED MODULE ) not allowed!" )
            ENDIF()
            set( libraryType ${argument} )
            CONTINUE()
        ENDIF()
        IF( argument STREQUAL "EXCLUDE_FROM_ALL" )
            SET( excludeFromAll ${argument} )
            CONTINUE()
        ENDIF()

        # 4.) ELSE the argument is a file name
        list( APPEND sourceFileNames "${argument}" )
    ENDFOREACH()
    UNSET( optionsEncountered )
    #message( "libraryType = ${libraryType}" )
    #message( "sourceFileNames = ${sourceFileNames}" )

    # call add_library or cuda_add_library now
    IF( alpaka_ACC_GPU_CUDA_ENABLE )
        ENABLE_LANGUAGE(CUDA)
        FOREACH( _file ${ARGN} )
            IF( ( ${_file} MATCHES "\\.cpp$" ) OR
                ( ${_file} MATCHES "\\.cxx$" ) OR
                ( ${_file} MATCHES "\\.cu$" )
            )
                SET_SOURCE_FILES_PROPERTIES(${_file} PROPERTIES LANGUAGE CUDA)
            ENDIF()
        ENDFOREACH()

        ADD_LIBRARY(
            ${libraryName}
            ${libraryType}
            ${excludeFromAll}
            ${optionArguments}
            ${sourceFileNames}
            )
    ELSE()
        #message( "add_library( ${libraryName} ${libraryType} ${excludeFromAll} ${sourceFileNames} )" )
        ADD_LIBRARY(
            ${libraryName}
            ${libraryType}
            ${excludeFromAll}
            ${sourceFileNames}
        )
    ENDIF()

    # UNSET variables (not sure if necessary)
    UNSET( libraryType )
    UNSET( sourceFileNames )
    UNSET( excludeFromAll )
    UNSET( optionArguments )
ENDMACRO()
