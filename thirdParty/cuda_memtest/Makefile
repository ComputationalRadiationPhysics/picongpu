#  Illinois Open Source License
#  #  
#  University of Illinois/NCSA
#  #  Open Source License
#  
#  #  Copyright © 2009,    University of Illinois.  All rights reserved.
#  
#  #  Developed by:
#  
#  #  Innovative Systems Lab  
#  National Center for Supercomputing Applications  
#  #  http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
#  
#  #  Permission is hereby granted, free of charge, to any person obtaining a copy of 
#  this software and associated documentation files (the "Software"), to deal with 
#  #  the Software without restriction, including without limitation the rights to use,
#  copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
#  #  Software, and to permit persons to whom the Software is furnished to do so, subject
#  to the following conditions:
#  #  
#  * Redistributions of source code must retain the above copyright notice, this list 
#  #  of conditions and the following disclaimers.
#
##  * Redistributions in binary form must reproduce the above copyright notice, this list
#  of conditions and the following disclaimers in the documentation and/or other materials
#  #  provided with the distribution.
#
##  * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
#  Applications, nor the names of its contributors may be used to endorse or promote products
#  #  derived from this Software without specific prior written permission.
#  
#  #  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  #  PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
#  #  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
#  DEALINGS WITH THE SOFTWARE.
#  #

#default, assume you are using NVIDIA GPUs
#default target is cuda_memtest
#TARGET=cuda_memtest ocl_memtest
TARGET=cuda_memtest
CUDA_INSTALL_PATH ?=/usr/local/cuda

#if you are using AMD GPUs, uncomment the following line and set the install path correctly 
#TARGET=ocl_memtest
AMD_INSTALL_PATH ?=/usr/local/ati-stream-sdk-v2.1-lnx64/


CC=gcc
CXX=g++
CUDACC=nvcc

CUDA_INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include 
CUDALIB := -L${CUDA_INSTALL_PATH}/lib64 -lcuda  -lcudart -lpthread
CFLAGS= -arch sm_13 -DSM_13 -O3
CFLAGS_SM10= -arch sm_10 -DSM_10 -O3
CFLAGS_SM20= -arch sm_20 -DSM_20 -O3
CUDA_SRC_FILES= cuda_memtest.cu misc.cpp tests.cu
CUDA_OBJS=  cuda_memtest.o misc.o tests.o

AMD_OPENCL_INC_DIR =${AMD_INSTALL_PATH}/include
AMD_OPENCL_LIB_DIR =${AMD_INSTALL_PATH}/lib/x86_64/
NVIDIA_OPENCL_INC_DIR=${CUDA_INCLUDES}
NVIDIA_OPENCL_LIB_DIR=${CUDA_INSTALL_PATH}/lib64
OCL_SRC_FILES= ocl_memtest.cpp ocl_tests.cpp 
OCL_OBJS= ocl_memtest.o ocl_tests.o
OCL_LDFLAGS= -lpthread -lOpenCL -L${AMD_OPENCL_LIB_DIR} -L${NVIDIA_OPENCL_LIB_DIR}

default: ${TARGET}
all: ${TARGET} cuda_memtest_sm10 cuda_memtest_sm20
.SUFFIXES: .o .cu .cpp
.cu.o:
	${CUDACC} -c ${CFLAGS}  ${CUDA_INCLUDES} -o $@ $<
misc.o: misc.cpp
	${CUDACC} -c ${CFLAGS}  ${CUDA_INCLUDES} -o $@ $<

ocl_memtest.o: ocl_memtest.cpp
	${CXX} -c ${CPPFLAGS} $< -I${NVIDIA_OPENCL_INC_DIR} -I${AMD_OPENCL_INC_DIR}
ocl_tests.o: ocl_tests.cpp
	${CXX} -c ${CPPFLAGS} $< -I${NVIDIA_OPENCL_INC_DIR} -I${AMD_OPENCL_INC_DIR}

ocl_memtest: ${OCL_OBJS}
	${CXX} -o $@ ${OCL_OBJS} ${OCL_LDFLAGS}


cuda_memtest: ${CUDA_OBJS}
	${CUDACC}  -o  $@ ${CUDA_OBJS} ${CUDALIB}	
cuda_memtest_sm10: ${SRCS}
	${CUDACC} -c ${CFLAGS_SM10}  ${CUDA_INCLUDES} -o tests.o tests.cu
	${CUDACC} -c ${CFLAGS_SM10}  ${CUDA_INCLUDES} -o misc.o misc.cpp
	${CUDACC} -c ${CFLAGS_SM10}  ${CUDA_INCLUDES} -o cuda_memtest.o cuda_memtest.cu
	${CUDACC}  -o  $@ ${CUDA_OBJS} ${CUDALIB}
cuda_memtest_sm20: ${SRCS}
	${CUDACC} -c ${CFLAGS_SM20}  ${CUDA_INCLUDES} -o tests.o tests.cu
	${CUDACC} -c ${CFLAGS_SM20}  ${CUDA_INCLUDES} -o misc.o misc.cpp
	${CUDACC} -c ${CFLAGS_SM20}  ${CUDA_INCLUDES} -o cuda_memtest.o cuda_memtest.cu
	${CUDACC}  -o  $@ ${CUDA_OBJS} ${CUDALIB}
clean:	
	rm -fr *.o ${TARGET} *~ *.linkinfo cuda_memtest_sm10 cuda_memtest_sm20
