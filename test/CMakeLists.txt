cmake_minimum_required(VERSION 3.14...3.22)

project(Examples LANGUAGES CXX)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/unit
    ${CMAKE_BINARY_DIR}/test/unit
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/multithreaded
    ${CMAKE_BINARY_DIR}/test/multithreaded
)

add_custom_target(
    mallocMCTests
    DEPENDS mallocMCUnitTests mallocMCMultithreadedTests
    COMMENT "Shortcut for building all tests"
)
