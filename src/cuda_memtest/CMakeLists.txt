cmake_minimum_required(VERSION 2.8)


# Projekt name
project (CUDA_memtest)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 


#set helper pathes to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{MPI_ROOT}" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}" "$ENV{HDF5_ROOT}")

#include cuda plugin for cmake
FIND_PACKAGE(CUDA REQUIRED)

########################################################################
#set options

##CUDA##

IF(NOT DISBALE_COMPILER_FLAGS_IN_SUBPROJECTS)
  SET(CUDA_ARCH sm_13 CACHE STRING "set GPU architecture" )
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH})

  IF(CUDA_SHOW_REGISTER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -Xptxas=-v -Xopencc=-LIST:source=on)
  ENDIF()

  IF(CUDA_KEEP_FILES)
    MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" --keep --keep-dir "nvcc_tmp")
  ENDIF()
ENDIF()

##END CUDA##


OPTION(CUDA_MEMTEST_RELEASE "disable all runtime asserts" ON)
IF(PIC_RELEASE)
    ADD_DEFINITIONS(-DNDEBUG)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler=-pthread")
ELSE(PIC_RELEASE)
   set(CMAKE_BUILD_TYPE Debug)
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g,-pthread")
ENDIF(PIC_RELEASE)
SET(CMAKE_CXX_FLAGS_DEFAULT "-Wall")


#########################################################################
# Configure include directories



set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${INCLUDE_DIRECTORIES})


cuda_add_executable(cuda_memtest
    tests.cu
    misc.cpp
    cuda_memtest.cu
)


target_link_libraries(cuda_memtest  ${LIBS} ${CUDA_CUDART_LIBRARY})



#### INSTALLATION #####
INSTALL(TARGETS cuda_memtest
         RUNTIME DESTINATION bin)