cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCMultithreadedTests LANGUAGES CXX)

# ---- Options ----

option(mallocMC_ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(mallocMC_TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tools.cmake)

# ---- Dependencies ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CPM_0.40.2.cmake)
CPMUsePackageLock(${CMAKE_CURRENT_LIST_DIR}/../../cmake/package-lock.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/add_controlled.cmake)

add_controlled("alpaka" REQUIRED PREFIX mallocMC)
add_controlled("Catch2" REQUIRED PREFIX mallocMC)

if(NOT TARGET mallocMC)
  if(mallocMC_TEST_INSTALLED_VERSION)
    find_package(mallocMC REQUIRED)
  else()
    CPMAddPackage(NAME mallocMC SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
  endif()
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
alpaka_add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} mallocMC::mallocMC alpaka::alpaka Catch2::Catch2WithMain)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# enable compiler warnings
if(NOT mallocMC_TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(
      mallocMC
      INTERFACE
      -Wall
      # nvcc generate C code which uses directives GCC complains about like this:
      #   warning: style of line directive is a GCC extension
      # So we can't use -pedantic here.
      $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wpedantic>
      -Wextra
      # Somehow, with the commandline that CMake composes nvcc misinterprets the flag
      # after -Werror as an argument to -Werror leading to errors like
      #   nvcc fatal   : Value '-Wpedantic' is not defined for option 'Werror'
      # So, we can't compile with -Werror for nvcc.
      $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>
    )
  elseif(MSVC)
    target_compile_options(mallocMC INTERFACE /W4 /WX)
  endif()
endif()

# ---- Add mallocMCTests ----

enable_testing()
add_test(${PROJECT_NAME} ${PROJECT_NAME})

# ---- code coverage ----

if(mallocMC_ENABLE_TEST_COVERAGE)
  target_compile_options(mallocMC INTERFACE -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(mallocMC INTERFACE -fprofile-arcs -ftest-coverage)
endif()
