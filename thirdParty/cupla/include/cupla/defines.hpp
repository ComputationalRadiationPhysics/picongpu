/* Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdint>

#include "cupla/namespace.hpp"

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED 1
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#   undef ALPAKA_ACC_GPU_CUDA_ENABLED
#   define ALPAKA_ACC_GPU_CUDA_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED 1
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#   undef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
#   define ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED 1
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#   undef ALPAKA_ACC_GPU_HIP_ENABLED
#   define ALPAKA_ACC_GPU_HIP_ENABLED 1
#endif

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
#   undef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
#   define ALPAKA_ACC_ANY_BT_OMP5_ENABLED 1
#endif

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
#   undef ALPAKA_ACC_ANY_BT_OACC_ENABLED
#   define ALPAKA_ACC_ANY_BT_OACC_ENABLED 1
#endif

#define CUPLA_NUM_SELECTED_DEVICES (                                           \
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED +                               \
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED  +                                 \
        ALPAKA_ACC_GPU_CUDA_ENABLED +                                          \
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED +                                   \
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED +                                   \
        ALPAKA_ACC_GPU_HIP_ENABLED +                                           \
        ALPAKA_ACC_ANY_BT_OMP5_ENABLED +                                       \
        ALPAKA_ACC_ANY_BT_OACC_ENABLED                                         \
)


#if( CUPLA_NUM_SELECTED_DEVICES == 0 )
    #error "there is no accelerator selected, please run `ccmake .` and select one"
#endif

#if( CUPLA_NUM_SELECTED_DEVICES > 2  )
    #error "please select at most two accelerators"
#endif

// count accelerators where the thread count must be one
#define CUPLA_NUM_SELECTED_THREAD_SEQ_DEVICES (                                \
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED +                                   \
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED                                     \
)

#define CUPLA_NUM_SELECTED_THREAD_PARALLEL_DEVICES (                           \
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED +                                  \
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED +                               \
        ALPAKA_ACC_GPU_CUDA_ENABLED +                                          \
        ALPAKA_ACC_GPU_HIP_ENABLED +                                           \
        ALPAKA_ACC_ANY_BT_OMP5_ENABLED +                                       \
        ALPAKA_ACC_ANY_BT_OACC_ENABLED                                         \
)

#if( CUPLA_NUM_SELECTED_THREAD_SEQ_DEVICES > 1 )
    #error "it is only alowed to select one thread sequential Alpaka accelerator"
#endif

#if( CUPLA_NUM_SELECTED_THREAD_PARALLEL_DEVICES > 1 )
    #error "it is only alowed to select one thread parallelized Alpaka accelerator"
#endif

#ifndef CUPLA_HEADER_ONLY_FUNC_SPEC
#   define CUPLA_HEADER_ONLY_FUNC_SPEC
#endif

/*! device compile flag
 *
 * Enabled if the compiler processes currently a separate compile path for the device code
 *
 * @attention value is always 0 for alpaka CPU accelerators
 *
 * Value is 1 if device path is compiled else 0
 */
#if defined(__CUDA_ARCH__) || ( defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__== 1 && defined(__HIP__) )
    #define CUPLA_DEVICE_COMPILE 1
#else
    #define CUPLA_DEVICE_COMPILE 0
#endif
