#  This file was copied from
#  https://github.com/TheLartians/PackageProject.cmake/blob/2b8d9b49d227f714092ffe5263966274dc863380/CMakeLists.txt
#
#  MIT License
#
# Copyright (c) 2020 Lars Melchior
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# /

cmake_minimum_required(VERSION 3.14)

set(PACKAGE_PROJECT_ROOT_PATH
    "${CMAKE_CURRENT_LIST_DIR}"
    CACHE INTERNAL "The path to the PackageProject directory"
)

function(packageProject)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)

  cmake_parse_arguments(
    PROJECT
    ""
    "NAME;VERSION;INCLUDE_DIR;INCLUDE_DESTINATION;BINARY_DIR;COMPATIBILITY;EXPORT_HEADER;VERSION_HEADER;NAMESPACE;DISABLE_VERSION_SUFFIX;ARCH_INDEPENDENT;INCLUDE_HEADER_PATTERN;CPACK;RUNTIME_DESTINATION"
    "DEPENDENCIES"
    ${ARGN}
  )

  # optional feature: TRUE or FALSE or UNDEFINED! These variables will then hold the respective
  # value from the argument list or be undefined if the associated one_value_keyword could not be
  # found.
  if(PROJECT_DISABLE_VERSION_SUFFIX)
    unset(PROJECT_VERSION_SUFFIX)
  else()
    set(PROJECT_VERSION_SUFFIX -${PROJECT_VERSION})
  endif()

  if(NOT DEFINED PROJECT_COMPATIBILITY)
    set(PROJECT_COMPATIBILITY AnyNewerVersion)
  endif()

  # we want to automatically add :: to our namespace, so only append if a namespace was given in the
  # first place we also provide an alias to ensure that local and installed versions have the same
  # name
  if(DEFINED PROJECT_NAMESPACE)
    if(PROJECT_CPACK)
      set(CPACK_PACKAGE_NAMESPACE ${PROJECT_NAMESPACE})
    endif()
    set(PROJECT_NAMESPACE ${PROJECT_NAMESPACE}::)
    add_library(${PROJECT_NAMESPACE}${PROJECT_NAME} ALIAS ${PROJECT_NAME})
  endif()

  if(DEFINED PROJECT_VERSION_HEADER OR DEFINED PROJECT_EXPORT_HEADER)
    set(PROJECT_VERSION_INCLUDE_DIR ${PROJECT_BINARY_DIR}/PackageProjectInclude)

    if(DEFINED PROJECT_EXPORT_HEADER)
      include(GenerateExportHeader)
      generate_export_header(
        ${PROJECT_NAME} EXPORT_FILE_NAME ${PROJECT_VERSION_INCLUDE_DIR}/${PROJECT_EXPORT_HEADER}
      )
    endif()

    if(DEFINED PROJECT_VERSION_HEADER)
      # clear previous matches
      unset(CMAKE_MATCH_1)
      unset(CMAKE_MATCH_3)
      unset(CMAKE_MATCH_5)
      unset(CMAKE_MATCH_7)

      string(REGEX MATCH "^([0-9]+)(\\.([0-9]+))?(\\.([0-9]+))?(\\.([0-9]+))?$" _
                   "${PROJECT_VERSION}"
      )

      set(PROJECT_VERSION_MAJOR ${CMAKE_MATCH_1})
      set(PROJECT_VERSION_MINOR ${CMAKE_MATCH_3})
      set(PROJECT_VERSION_PATCH ${CMAKE_MATCH_5})
      set(PROJECT_VERSION_TWEAK ${CMAKE_MATCH_7})

      if(NOT DEFINED PROJECT_VERSION_MAJOR)
        set(PROJECT_VERSION_MAJOR "0")
      endif()
      if(NOT DEFINED PROJECT_VERSION_MINOR)
        set(PROJECT_VERSION_MINOR "0")
      endif()
      if(NOT DEFINED PROJECT_VERSION_PATCH)
        set(PROJECT_VERSION_PATCH "0")
      endif()
      if(NOT DEFINED PROJECT_VERSION_TWEAK)
        set(PROJECT_VERSION_TWEAK "0")
      endif()

      string(TOUPPER ${PROJECT_NAME} UPPERCASE_PROJECT_NAME)
      # ensure that the generated macro does not include invalid characters
      string(REGEX REPLACE [^a-zA-Z0-9] _ UPPERCASE_PROJECT_NAME ${UPPERCASE_PROJECT_NAME})
      configure_file(
        ${PACKAGE_PROJECT_ROOT_PATH}/version.h.in
        ${PROJECT_VERSION_INCLUDE_DIR}/${PROJECT_VERSION_HEADER} @ONLY
      )
    endif()

    get_target_property(target_type ${PROJECT_NAME} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
      set(VISIBILITY INTERFACE)
    else()
      set(VISIBILITY PUBLIC)
    endif()
    target_include_directories(
      ${PROJECT_NAME} ${VISIBILITY} "$<BUILD_INTERFACE:${PROJECT_VERSION_INCLUDE_DIR}>"
    )
    install(
      DIRECTORY ${PROJECT_VERSION_INCLUDE_DIR}/
      DESTINATION ${PROJECT_INCLUDE_DESTINATION}
      COMPONENT "${PROJECT_NAME}_Development"
    )
  endif()

  set(wbpvf_extra_args "")
  if(NOT DEFINED PROJECT_ARCH_INDEPENDENT)
    get_target_property(target_type "${PROJECT_NAME}" TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
      set(PROJECT_ARCH_INDEPENDENT YES)
    endif()
  endif()

  if(PROJECT_ARCH_INDEPENDENT)
    set(wbpvf_extra_args ARCH_INDEPENDENT)
    # install to architecture independent (share) directory
    set(INSTALL_DIR_FOR_CMAKE_CONFIGS ${CMAKE_INSTALL_DATADIR})
  else()
    # if x32 or multilib->x32 , install to (lib) directory. if x64, install to (lib64) directory
    set(INSTALL_DIR_FOR_CMAKE_CONFIGS ${CMAKE_INSTALL_LIBDIR})
  endif()

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ${PROJECT_COMPATIBILITY} ${wbpvf_extra_args}
  )

  # set default runtime install subdirectory (RUNTIME_DESTINATION)
  if(NOT DEFINED PROJECT_RUNTIME_DESTINATION)
    set(PROJECT_RUNTIME_DESTINATION ${PROJECT_NAME}${PROJECT_VERSION_SUFFIX})
  endif()

  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_RUNTIME_DESTINATION}
            COMPONENT "${PROJECT_NAME}_Runtime"
            NAMELINK_COMPONENT "${PROJECT_NAME}_Development"
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_RUNTIME_DESTINATION}
            COMPONENT "${PROJECT_NAME}_Development"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_RUNTIME_DESTINATION}
            COMPONENT "${PROJECT_NAME}_Runtime"
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_RUNTIME_DESTINATION}
           COMPONENT "${PROJECT_NAME}_Runtime"
    PUBLIC_HEADER DESTINATION ${PROJECT_INCLUDE_DESTINATION} COMPONENT "${PROJECT_NAME}_Development"
    INCLUDES
    DESTINATION "${PROJECT_INCLUDE_DESTINATION}"
  )

  set("${PROJECT_NAME}_INSTALL_CMAKEDIR"
      "${INSTALL_DIR_FOR_CMAKE_CONFIGS}/cmake/${PROJECT_NAME}${PROJECT_VERSION_SUFFIX}"
      CACHE PATH "CMake package config location relative to the install prefix"
  )

  mark_as_advanced("${PROJECT_NAME}_INSTALL_CMAKEDIR")

  configure_file(
    ${PACKAGE_PROJECT_ROOT_PATH}/Config.cmake.in
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY
  )

  install(
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
    NAMESPACE ${PROJECT_NAMESPACE}
    COMPONENT "${PROJECT_NAME}_Development"
  )

  install(
    FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
          "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
    COMPONENT "${PROJECT_NAME}_Development"
  )

  if(NOT DEFINED PROJECT_INCLUDE_HEADER_PATTERN)
    set(PROJECT_INCLUDE_HEADER_PATTERN "*")
  endif()

  if(PROJECT_INCLUDE_DESTINATION AND PROJECT_INCLUDE_DIR)
    install(
      DIRECTORY ${PROJECT_INCLUDE_DIR}/
      DESTINATION ${PROJECT_INCLUDE_DESTINATION}
      COMPONENT "${PROJECT_NAME}_Development"
      FILES_MATCHING
      PATTERN "${PROJECT_INCLUDE_HEADER_PATTERN}"
    )
  endif()

  set(${PROJECT_NAME}_VERSION
      ${PROJECT_VERSION}
      CACHE INTERNAL ""
  )

  if(PROJECT_CPACK)
    if(CPACK_PACKAGE_NAMESPACE)
      set(CPACK_PACKAGE_NAME ${CPACK_PACKAGE_NAMESPACE}-${PROJECT_NAME})
    else()
      set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    endif()
    if(NOT CPACK_PACKAGE_DESCRIPTION_SUMMARY)
      set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
    endif()
    if(NOT CPACK_PACKAGE_HOMEPAGE_URL)
      set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_HOMEPAGE_URL}")
    endif()
    set(CPACK_VERBATIM_VARIABLES YES)
    set(CPACK_THREADS 0)
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
      set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    endif()

    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    )

    include(CPack)
  endif()
endfunction()
