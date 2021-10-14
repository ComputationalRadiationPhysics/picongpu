################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.16.0)


################################################################################
# Project
################################################################################

project(CUDA_memtest LANGUAGES CXX CUDA)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

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

if(SAME_NVCC_FLAGS_IN_SUBPROJECTS)
    if(CUDA_SHOW_CODELINES)
        target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx -lineinfo>)
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    if(CUDA_SHOW_REGISTER)
        target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas=-v>)
    endif()

    if(CUDA_KEEP_FILES)
        target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
    endif()
endif()

################################################################################
# Find NVML
################################################################################

set(GPU_DEPLOYMENT_KIT_ROOT_DIR "$ENV{GDK_ROOT}")
find_package(NVML)


################################################################################
# Find PThreads
################################################################################

if(NOT MSVC)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif()

find_package(Threads REQUIRED)


################################################################################
# Warnings
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


################################################################################
# Compile & Link cuda_memtest
################################################################################

set_source_files_properties(tests.cu cuda_memtest.cu misc.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cuda_memtest
        tests.cu
        misc.cpp
        cuda_memtest.cu
        )

target_link_libraries("cuda_memtest" INTERFACE CUDA::cudart)
target_link_libraries(cuda_memtest INTERFACE CUDA::cuda_driver)


if(NOT MSVC)
    target_link_libraries(cuda_memtest PRIVATE Threads::Threads)
endif()


if(NVML_FOUND)
    target_link_libraries(cuda_memtest PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(cuda_memtest PRIVATE "ENABLE_NVML=1")
else()
    target_compile_definitions(cuda_memtest PRIVATE "ENABLE_NVML=0")
endif()

################################################################################
# Build type (debug, release)
################################################################################

option(CUDA_MEMTEST_RELEASE "disable all runtime asserts" ON)
if(CUDA_MEMTEST_RELEASE)
    target_compile_definitions(cuda_memtest PRIVATE -DNDEBUG)
else(CUDA_MEMTEST_RELEASE)
    set(CMAKE_BUILD_TYPE Debug)
    target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-g>)
endif(CUDA_MEMTEST_RELEASE)

################################################################################
# Install cuda_memtest
################################################################################

install(TARGETS cuda_memtest
        RUNTIME DESTINATION bin)
