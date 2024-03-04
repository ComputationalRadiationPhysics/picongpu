#
# Copyright 2014-2019 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

#------------------------------------------------------------------------------
# Gets all recursive files with the given ending in the given directory and recursively below.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
FUNCTION(append_recursive_files In_RootDir In_FileExtension Out_FilePathsListVariableName)
    #MESSAGE("In_RootDir: ${In_RootDir}")
    #MESSAGE("In_FileExtension: ${In_FileExtension}")
    #MESSAGE("Out_FilePathsListVariableName: ${Out_FilePathsListVariableName}")
    # Get all recursive files.
    FILE(
        GLOB_RECURSE
        relativeFilePathsList
        "${In_RootDir}/*.${In_FileExtension}")
    #MESSAGE( "relativeFilePathsList: ${relativeFilePathsList}" )
    # Set the return value (append it to the value in the parent scope).
    SET(
        ${Out_FilePathsListVariableName}
        "${${Out_FilePathsListVariableName}}" "${relativeFilePathsList}"
        PARENT_SCOPE)
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all recursive relative subdirectories.
FUNCTION(append_recursive_relative_subdirs In_RootDir Out_RecursiveRelativeSubDirsVariableName)
    #MESSAGE("In_RootDir: ${In_RootDir}")
    # Get all the recursive files with their relative paths.
    FILE(
        GLOB_RECURSE
        recursiveRelativeFiles
        RELATIVE "${In_RootDir}/" "${In_RootDir}/*")
    #MESSAGE("recursiveRelativeFiles: ${recursiveRelativeFiles}")

    # Get the paths to all the recursive files.
    # Create empty list for the case of no subdirectories being present.
    SET(recursiveRelativeSubDirs)
    FOREACH(
        relativeFilePath
        IN LISTS recursiveRelativeFiles)
        GET_FILENAME_COMPONENT(
            relativeSubDir
            "${relativeFilePath}"
            PATH)
        LIST(
            APPEND
            recursiveRelativeSubDirs
            "${relativeSubDir}")
    ENDFOREACH()
    #MESSAGE("recursiveRelativeSubDirs: ${recursiveRelativeSubDirs}")

    # If the list is not empty.
    LIST(
        LENGTH
        recursiveRelativeSubDirs
        recursiveRelativeSubDirsLength)
    IF("${recursiveRelativeSubDirsLength}")
        # Remove duplicates from the list.
        LIST(
            REMOVE_DUPLICATES
            recursiveRelativeSubDirs)
        #MESSAGE("recursiveRelativeSubDirs: ${recursiveRelativeSubDirs}")

        # Set the return value (append it to the value in the parent scope).
        #MESSAGE("Out_RecursiveRelativeSubDirsVariableName: ${Out_RecursiveRelativeSubDirsVariableName}")
        SET(
            ${Out_RecursiveRelativeSubDirsVariableName}
            "${${Out_RecursiveRelativeSubDirsVariableName}}" "${recursiveRelativeSubDirs}"
            PARENT_SCOPE)
    ENDIF()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Groups the files in the same way the directories are structured.
FUNCTION(add_recursive_files_to_src_group In_RootDir In_SrcGroupIgnorePrefix In_FileExtension)
    #MESSAGE("In_RootDir: ${In_RootDir}")
    #MESSAGE("In_SrcGroupIgnorePrefix: ${In_SrcGroupIgnorePrefix}")
    #MESSAGE("In_FileExtension: ${In_FileExtension}")
    SET(recursiveRelativeSubDirs)
    # Get all recursive subdirectories.
    append_recursive_relative_subdirs(
        "${In_RootDir}"
        recursiveRelativeSubDirs)
    #MESSAGE("recursiveRelativeSubDirs: ${recursiveRelativeSubDirs}")

    # For the folder itself and each sub-folder...
    FOREACH(
        currentRelativeSubDir
        IN
        LISTS recursiveRelativeSubDirs
        ITEMS "")
        # Appended the current subdirectory.
        IF(currentRelativeSubDir STREQUAL "")
            SET(
                currentSubDir
                "${In_RootDir}")
        ELSE()
            SET(
                currentSubDir
                "${In_RootDir}/${currentRelativeSubDir}")
        ENDIF()
        #MESSAGE("currentSubDir: ${currentSubDir}")
        # Get all the files in this sub-folder.
        SET(
            wildcardFilePath
            "${currentSubDir}/*.${In_FileExtension}")
        #MESSAGE("wildcardFilePath: ${wildcardFilePath}")
        FILE(
            GLOB
            filesInSubDirList
            "${wildcardFilePath}")
        #MESSAGE("filesInSubDirList: ${filesInSubDirList}")

        LIST(
            LENGTH
            filesInSubDirList
            filesInSubDirListLength)
        IF("${filesInSubDirListLength}")
            # Group the include files into a project sub-folder analogously to the filesystem hierarchy.
            SET(
                groupExpression
                "${currentSubDir}")
            #MESSAGE("groupExpression: ${groupExpression}")
            # Remove the parent directory from the path.
            # NOTE: This is not correct because it does not only replace at the beginning of the string.
            #  "STRING(REGEX REPLACE" would be correct if there was an easy way to escape arbitrary strings.
            STRING(
                REPLACE "${In_SrcGroupIgnorePrefix}" ""
                groupExpression
                "${groupExpression}")
            # Remove leading slash.
            STRING(
                REGEX REPLACE "^/" ""
                groupExpression
                "${groupExpression}")
            #MESSAGE("groupExpression: ${groupExpression}")
            # Replace the directory separators in the path to build valid grouping expressions.
            STRING(
                REPLACE "/" "\\"
                groupExpression
                "${groupExpression}")
            #MESSAGE("groupExpression: ${groupExpression}")
            SOURCE_GROUP(
                "${groupExpression}"
                FILES ${filesInSubDirList})
        ENDIF()
    ENDFOREACH()
ENDFUNCTION()

#------------------------------------------------------------------------------
# Gets all files with the given ending in the given directory.
# Groups the files in the same way the directories are structured.
# This makes adding files easier because we do not have to update a list each time a file is added but this prevents CMake from detecting if it should be rerun!
FUNCTION(append_recursive_files_add_to_src_group In_RootDir In_SrcGroupIgnorePrefix In_FileExtension Out_FilePathsListVariableName)
    #MESSAGE("In_RootDir: ${In_RootDir}")
    #MESSAGE("In_SrcGroupIgnorePrefix: ${In_SrcGroupIgnorePrefix}")
    #MESSAGE("In_FileExtension: ${In_FileExtension}")
    #MESSAGE("Out_FilePathsListVariableName: ${Out_FilePathsListVariableName}")
    # We have to use a local variable and give it to the parent because append_recursive_files only gives it to our scope but not the one calling this function.
    SET(
        allFilePathsList
        "${${Out_FilePathsListVariableName}}")
    append_recursive_files(
        "${In_RootDir}"
        "${In_FileExtension}"
        allFilePathsList)
    #MESSAGE( "allFilePathsList: ${allFilePathsList}" )
    # Set the return value (append it to the value in the parent scope).
    SET(
        ${Out_FilePathsListVariableName}
        "${${Out_FilePathsListVariableName}}" "${allFilePathsList}"
        PARENT_SCOPE)

    add_recursive_files_to_src_group(
        "${In_RootDir}"
        "${In_SrcGroupIgnorePrefix}"
        "${In_FileExtension}")
ENDFUNCTION()

#------------------------------------------------------------------------------
# void list_add_prefix(string In_Prefix, list<string>* In_ListVariableName);
# - returns The In_ListVariableName with In_Prefix prepended to all items.
# - original list is modified
FUNCTION(list_add_prefix In_Prefix In_ListVariableName)
    SET(local_list)

    FOREACH(
        item
        IN LISTS ${In_ListVariableName})
        IF(NOT "${item}" STREQUAL "")
            LIST(
                APPEND
                local_list
                "${In_Prefix}${item}")
        ENDIF()
    ENDFOREACH()

    SET(
        ${In_ListVariableName}
        "${local_list}"
        PARENT_SCOPE)
ENDFUNCTION()
