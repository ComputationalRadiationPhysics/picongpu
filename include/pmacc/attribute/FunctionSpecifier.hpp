/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cupla/types.hpp>


#define HDINLINE ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
#define DINLINE ALPAKA_FN_ACC ALPAKA_FN_INLINE
#define DEVICEONLY ALPAKA_FN_ACC
#define HINLINE ALPAKA_FN_HOST ALPAKA_FN_INLINE

/**
 * CUDA architecture version (aka PTX ISA level)
 * 0 for host compilation
 */
#ifndef __CUDA_ARCH__
#    define PMACC_CUDA_ARCH 0
#else
#    define PMACC_CUDA_ARCH __CUDA_ARCH__
#endif

/** PMacc global identifier for CUDA kernel */
#define PMACC_GLOBAL_KEYWORD DINLINE

/*
 * Disable nvcc warning:
 * calling a __host__ function from __host__ __device__ function.
 *
 * Usage:
 * PMACC_NO_NVCC_HDWARNING
 * HDINLINE function_declaration()
 *
 * It is not possible to disable the warning for a __host__ function
 * if there are calls of virtual functions inside. For this case use a wrapper
 * function.
 * WARNING: only use this method if there is no other way to create runable code.
 * Most cases can solved by #ifdef __CUDA_ARCH__ or #ifdef __CUDACC__.
 */
#if defined(__CUDACC__)
#    define PMACC_NO_NVCC_HDWARNING _Pragma("hd_warning_disable")
#else
#    define PMACC_NO_NVCC_HDWARNING
#endif
