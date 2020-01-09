/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, René Widera
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

#include <alpaka/math/trunc/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA trunc.
        class TruncCudaBuiltIn
        {
        public:
            using TruncBase = TruncCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA trunc trait specialization.
            template<
                typename TArg>
            struct Trunc<
                TruncCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto trunc(
                    TruncCudaBuiltIn const & trunc_ctx,
                    TArg const & arg)
                -> decltype(::trunc(arg))
                {
                    alpaka::ignore_unused(trunc_ctx);
                    return ::trunc(arg);
                }
            };
            //! The CUDA trunc float specialization.
            template<>
            struct Trunc<
                TruncCudaBuiltIn,
                float>
            {
                __device__ static auto trunc(
                    TruncCudaBuiltIn const & trunc_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(trunc_ctx);
                    return ::truncf(arg);
                }
            };
        }
    }
}

#endif
