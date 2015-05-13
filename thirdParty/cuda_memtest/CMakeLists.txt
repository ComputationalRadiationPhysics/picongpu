################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.10)


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
# Find CUDA 
################################################################################

find_package(CUDA REQUIRED)

if(SAME_NVCC_FLAGS_IN_SUBPROJECTS)
  set(CUDA_ARCH sm_13 CACHE STRING "set GPU architecture" )
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH})

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


################################################################################
# Build type (debug, release)
################################################################################

option(CUDA_MEMTEST_RELEASE "disable all runtime asserts" ON)
if(PIC_RELEASE)
    add_definitions(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
else(PIC_RELEASE)
    set(CMAKE_BUILD_TYPE Debug)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g,-pthread")
endif(PIC_RELEASE)


################################################################################
# Warnings
################################################################################

set(CMAKE_CXX_FLAGS_DEFAULT "-Wall")


################################################################################
# Configure include directories
################################################################################

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${INCLUDE_DIRECTORIES})


################################################################################
# Compile & Link cuda_memtest
################################################################################

cuda_add_executable(cuda_memtest
    tests.cu
    misc.cpp
    cuda_memtest.cu
)

target_link_libraries(cuda_memtest ${LIBS} ${CUDA_CUDART_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# Install cuda_memtest
################################################################################

install(TARGETS cuda_memtest
         RUNTIME DESTINATION bin)
