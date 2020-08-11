project(mallocMC LANGUAGES CXX)
cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
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
if(${mallocMC_CATCH2_PROVIDER} STREQUAL "intern")
    add_library(Catch2::Catch2 INTERFACE IMPORTED)
    target_include_directories(Catch2::Catch2 INTERFACE ${CMAKE_CURRENT_LIST_DIR}/thirdParty/catch2/include)
else()
    find_package(Catch2 CONFIG REQUIRED)
endif()

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

alpaka_add_executable(VerifyHeap EXCLUDE_FROM_ALL tests/verify_heap.cpp tests/verify_heap_config.hpp)
target_include_directories(VerifyHeap PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
target_link_libraries(VerifyHeap PUBLIC alpaka::alpaka warnings)

if (CUDA_VERSION VERSION_LESS 10.2) # TODO(bgruber): I do not know exactly where it breaks. 9.1 does not work, 10.2 works
    # the catch2 main needs to be in a non-CUDA file before CUDA 10.2, because nvcc fails to compile the catch2 header
    # TODO: merge the test_main back into the tests exe, once CUDA 10.2 is the minimum version
    add_library(tests_main EXCLUDE_FROM_ALL tests/main.cpp)
    target_include_directories(tests_main PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
    target_link_libraries(tests_main PUBLIC Catch2::Catch2 warnings)

    alpaka_add_executable(tests EXCLUDE_FROM_ALL tests/dimensions.cpp tests/policies.cpp)
    target_include_directories(tests PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
    target_link_libraries(tests PUBLIC tests_main alpaka::alpaka Catch2::Catch2 warnings)
else()
    alpaka_add_executable(tests EXCLUDE_FROM_ALL tests/main.cpp tests/dimensions.cpp tests/policies.cpp)
    target_include_directories(tests PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/include)
    target_link_libraries(tests PUBLIC alpaka::alpaka Catch2::Catch2 warnings)
endif()

add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example03 VerifyHeap)
