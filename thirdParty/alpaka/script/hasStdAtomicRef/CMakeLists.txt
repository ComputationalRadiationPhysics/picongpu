cmake_minimum_required(VERSION 3.19)

project(hasStdAtomicRef LANGUAGES CXX)

enable_testing()

add_executable(${CMAKE_PROJECT_NAME})
target_sources(${CMAKE_PROJECT_NAME}
    PRIVATE
    main.cpp
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
)
add_test(NAME ${CMAKE_PROJECT_NAME} COMMAND ${CMAKE_PROJECT_NAME})
