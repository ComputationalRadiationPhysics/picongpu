/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Valentin Gehrke
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/rsqrt/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA rsqrt.
        class RsqrtCudaBuiltIn
        {
        public:
            using RsqrtBase = RsqrtCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA rsqrt trait specialization.
            template<
                typename TArg>
            struct Rsqrt<
                RsqrtCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                __device__ static auto rsqrt(
                    RsqrtCudaBuiltIn const & rsqrt_ctx,
                    TArg const & arg)
                -> decltype(::rsqrt(arg))
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrt(arg);
                }
            };
            //! The CUDA rsqrt float specialization.
            template<>
            struct Rsqrt<
                RsqrtCudaBuiltIn,
                float>
            {
                __device__ static auto rsqrt(
                    RsqrtCudaBuiltIn const & rsqrt_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrtf(arg);
                }
            };
        }
    }
}

#endif
