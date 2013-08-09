/**
 * Copyright 2013 Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#ifndef KERNELEVENTS_H
#define KERNELEVENTS_H

#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "eventSystem/EventSystem.hpp"

namespace PMacc
{

/*if this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if (PMACC_SYNC_KERNEL  == 1)
    #define CUDA_CHECK_KERNEL_MSG(...)  CUDA_CHECK_MSG(__VA_ARGS__)
#else
    /*no synchronize and check of kernel calls*/
    #define CUDA_CHECK_KERNEL_MSG(...)  ;
#endif

/**
 * Returns number of args... arguments.
 *
 * Can only count values of ... which can be casted to int type.
 *
 * @param ... arguments
 */
#define PMACC_NUMARGS(...)  (sizeof((int[]){0, ##__VA_ARGS__})/sizeof(int)-1)

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PMACC_CUDAPARAMS(...) (__VA_ARGS__);  \
        CUDA_CHECK_KERNEL_MSG(cudaThreadSynchronize(),"Crash after kernel call"); \
        taskKernel->activateChecks();              \
        CUDA_CHECK_KERNEL_MSG(cudaThreadSynchronize(),"Crash after kernel activation"); \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param grid sizes of grid on gpu
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PMACC_CUDAKERNELCONFIG(grid,block,...) <<<(grid),(block),     \
    PMACC_NUMARGS(__VA_ARGS__)==1?__VA_ARGS__:0,                      \
    taskKernel->getCudaStream()>>> PMACC_CUDAPARAMS

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param kernelname name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __cudaKernel(kernelname) {    \
    CUDA_CHECK_KERNEL_MSG(cudaThreadSynchronize(),"Crash before kernel call"); \
    TaskKernel *taskKernel =  Factory::getInstance().createTaskKernel(#kernelname);  \
    kernelname PMACC_CUDAKERNELCONFIG

}


#endif //KERNELEVENTS_H

