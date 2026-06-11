/*
 * SPDX-FileCopyrightText: Felix Schmitt, Heiko Burau, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <alpaka/alpaka.hpp>


/** Function qualifiers
 *
 * Function qualifier should stay left hand side of keyword e.g. 'static', 'constexpr', 'explicit' or the return type
 * definition.
 * @{
 */
#define HDINLINE ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
#define DINLINE ALPAKA_FN_ACC ALPAKA_FN_INLINE
#define DEVICEONLY ALPAKA_FN_ACC
#define HINLINE ALPAKA_FN_HOST ALPAKA_FN_INLINE
/** @} */

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
