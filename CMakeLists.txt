cmake_minimum_required(VERSION 3.14...3.22)

# ---- Project ----

project(
  mallocMC
  VERSION 3.0.0
  LANGUAGES CXX
)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM_0.40.2.cmake)
CPMUsePackageLock(${CMAKE_CURRENT_LIST_DIR}/cmake/package-lock.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/add_controlled.cmake)

# PackageProject.cmake will be used to make our target installable
add_controlled("PackageProject.cmake" REQUIRED)
add_controlled("alpaka" REQUIRED)

# ---- Create library ----

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
add_library(${PROJECT_NAME} INTERFACE)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

option(mallocMC_BUILD_TESTING "Turn on/off building the tests" OFF)
if(mallocMC_BUILD_TESTING)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/tools.cmake)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test ${CMAKE_BINARY_DIR}/test)
endif()

option(mallocMC_BUILD_EXAMPLES "Turn on/off building the examples" OFF)
if(mallocMC_BUILD_EXAMPLES)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/tools.cmake)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples ${CMAKE_BINARY_DIR}/examples)
endif()

# ---- Create an installable target ----
# this allows users to install and find the library via `find_package()`.

# the location where the project's version header will be placed should match the project's regular
# header paths
string(TOLOWER ${PROJECT_NAME}/version.hpp VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
)
