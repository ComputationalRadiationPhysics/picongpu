################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.5)


################################################################################
# Project
################################################################################

project(CUDA_memtest)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 

# set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}")

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


################################################################################
# Find CUDA
################################################################################

find_package(CUDA REQUIRED)

if(SAME_NVCC_FLAGS_IN_SUBPROJECTS)
  set(CUDA_ARCH sm_13 CACHE STRING "set GPU architecture" )
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH})
  if(CUDA_ARCH STREQUAL "sm_10")
    add_definitions(-DSM_10=1)
  endif()

  if(CUDA_SHOW_CODELINES)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --source-in-ptx -Xcompiler -rdynamic -lineinfo)
    set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
  endif(CUDA_SHOW_CODELINES)

  if(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v)
  endif()

  if(CUDA_KEEP_FILES)
    make_directory("${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
  endif()
endif()


################################################################################
# Find NVML
################################################################################

set(GPU_DEPLOYMENT_KIT_ROOT_DIR "$ENV{GDK_ROOT}")
find_package(NVML)

if(NVML_FOUND)
    include_directories(${NVML_INCLUDE_DIR})
    list(APPEND LIBS ${NVML_LIBRARY})
    add_definitions(-DENABLE_NVML=1)
else()
    add_definitions(-DENABLE_NVML=0)
endif()


################################################################################
# Find PThreads
################################################################################

find_package(Threads REQUIRED)
list(APPEND LIBS ${CMAKE_THREAD_LIBS_INIT})

# work around for Hypnos (Ubuntu 14.04) with gcc 4.8.2
if(NOT CMAKE_THREAD_LIBS_INIT)
    list(APPEND LIBS "-pthread")
endif()


################################################################################
# Build type (debug, release)
################################################################################

option(CUDA_MEMTEST_RELEASE "disable all runtime asserts" ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-pthread")
if(CUDA_MEMTEST_RELEASE)
    add_definitions(-DNDEBUG)
else(CUDA_MEMTEST_RELEASE)
    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
endif(CUDA_MEMTEST_RELEASE)


################################################################################
# Warnings
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


################################################################################
# Compile & Link cuda_memtest
################################################################################

cuda_add_executable(cuda_memtest
    tests.cu
    misc.cpp
    cuda_memtest.cu
)

target_link_libraries(cuda_memtest ${LIBS} ${CUDA_CUDART_LIBRARY})


################################################################################
# Install cuda_memtest
################################################################################

install(TARGETS cuda_memtest
        RUNTIME DESTINATION bin)
