project(mallocMC LANGUAGES CXX)
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# find alpaka
set(mallocMC_ALPAKA_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE mallocMC_ALPAKA_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(mallocMC_ALPAKA_PROVIDER)
if(${mallocMC_ALPAKA_PROVIDER} STREQUAL "intern")
  set(alpaka_BUILD_EXAMPLES OFF)
  set(BUILD_TESTING OFF)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/alpaka ${CMAKE_BINARY_DIR}/alpaka)
else()
  find_package(alpaka HINTS $ENV{ALPAKA_ROOT})
endif()

if(NOT TARGET alpaka::alpaka)
  message(FATAL "Required mallocMC dependency alpaka could not be found!")
endif()

# Catch2
set(mallocMC_CATCH2_PROVIDER "intern" CACHE STRING "Select which Catch2 is used")
set_property(CACHE mallocMC_CATCH2_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(mallocMC_CATCH2_PROVIDER)

# for installation, just copy include folder to install folder
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/include/."
    DESTINATION include
)

# warnings
add_library(warnings INTERFACE)
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(warnings INTERFACE -Wall -Wshadow -Wno-unknown-pragmas -Wextra -Wno-unused-parameter -Wno-unused-local-typedefs)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  target_compile_options(warnings INTERFACE -Wall -Wshadow)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  target_compile_options(warnings INTERFACE -Minform=inform)
endif()

# Executables
file(GLOB_RECURSE headers src/include/**)
add_custom_target(mallocMCIde SOURCES ${headers}) # create a target with the header files for IDE projects
source_group(TREE ${CMAKE_CURRENT_LIST_DIR}/src/include FILES ${headers})

alpaka_add_executable(mallocMC_Example01 EXCLUDE_FROM_ALL examples/mallocMC_example01.cpp)
target_include_directories(mallocMC_Example01 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
target_link_libraries(mallocMC_Example01 PUBLIC alpaka::alpaka warnings)

alpaka_add_executable(mallocMC_Example03 EXCLUDE_FROM_ALL examples/mallocMC_example03.cpp)
target_include_directories(mallocMC_Example03 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
target_link_libraries(mallocMC_Example03 PUBLIC alpaka::alpaka warnings)

add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example03)

if(${mallocMC_CATCH2_PROVIDER} STREQUAL "intern")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)
  include(Catch)
else()
  # get Catch2 v3 and build it from source with the same C++ standard as the tests
  Include(FetchContent)
  FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.7.1)
  FetchContent_MakeAvailable(Catch2)
  target_compile_features(Catch2 PUBLIC cxx_std_20)
  include(Catch)

  # hide Catch2 cmake variables by default in cmake gui
  get_cmake_property(variables VARIABLES)
  foreach (var ${variables})
    if (var MATCHES "^CATCH_")
      mark_as_advanced(${var})
    endif()
  endforeach()
endif()

file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/tests/*/*.cpp")
alpaka_add_executable(tests EXCLUDE_FROM_ALL ${testSources})
catch_discover_tests(tests)
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/tests" FILES ${testSources})
target_compile_features(tests PRIVATE cxx_std_20)
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
target_link_libraries(tests PRIVATE alpaka::alpaka Catch2::Catch2WithMain)
