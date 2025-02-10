cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCExampleVectorAdd LANGUAGES CXX)

# --- Import tools ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tools.cmake)

# ---- Dependencies ----

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/CPM_0.40.2.cmake)
CPMUsePackageLock(${CMAKE_CURRENT_LIST_DIR}/../../cmake/package-lock.cmake)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/add_controlled.cmake)

add_controlled("alpaka" REQUIRED PREFIX mallocMC)

if(NOT TARGET mallocMC)
  CPMAddPackage(NAME mallocMC SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
endif()

# ---- Create standalone executable ----

alpaka_add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        OUTPUT_NAME ${PROJECT_NAME}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_link_libraries(${PROJECT_NAME} mallocMC::mallocMC alpaka::alpaka)
