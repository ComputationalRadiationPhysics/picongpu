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

# ---- Options ----

option(mallocMC_BUILD_TESTING "Turn on/off building the tests" OFF)
option(mallocMC_BUILD_EXAMPLES "Turn on/off building the examples" OFF)
if (mallocMC_BUILD_TESTING OR mallocMC_BUILD_EXAMPLES)
  enable_testing()
endif()
if (mallocMC_BUILD_TESTING)
  set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(${CMAKE_CURRENT_LIST_DIR}/cmake/PackageProject.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM_0.40.2.cmake)
CPMUsePackageLock(${CMAKE_CURRENT_LIST_DIR}/cmake/package-lock.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/add_controlled.cmake)

# PackageProject.cmake will be used to make our target installable
add_controlled("alpaka" REQUIRED)


# ---- Create library ----

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
add_library(${PROJECT_NAME} INTERFACE)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  add_controlled("Gallatin")

  if (TARGET gallatin::gallatin)
    set(mallocMC_HAS_Gallatin_AVAILABLE YES)
  else()
    set(mallocMC_HAS_Gallatin_AVAILABLE NO)
  endif()

  # Gallatin needs some fairly recent compute capability from CUDA.
  # CMake defaults to taking the oldest supported by the device
  # (https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)
  # which can be too old. This leads to compilation errors along the lines of
  #
  #  error: no instance of overloaded function "atomicCAS" matches the argument list
  #         argument types are: (unsigned short *, unsigned short, unsigned short)
  #
  # because this overload was only added later (apparently?).

  if ("${CMAKE_CUDA_ARCHITECTURES}" LESS 70)
    message(
        WARNING
        "CUDA architecture detected is too old: ${CMAKE_CUDA_ARCHITECTURES}. "
        "If the architecture set is too old, this can lead to compilation errors with Gallatin. "
        "If Gallatin is needed, please set CMAKE_CUDA_ARCHITECTURES to the correct value >= 70."
      )
    set(mallocMC_HAS_Gallatin_AVAILABLE NO)
  endif()

  if (mallocMC_HAS_Gallatin_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} INTERFACE gallatin)
    target_compile_definitions(${PROJECT_NAME} INTERFACE mallocMC_HAS_Gallatin_AVAILABLE)
  endif()
endif()

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)
target_link_libraries(${PROJECT_NAME} INTERFACE alpaka::alpaka)


if(mallocMC_BUILD_TESTING)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/tools.cmake)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test ${CMAKE_BINARY_DIR}/test)
endif()

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
