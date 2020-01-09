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

#include <alpaka/math/sin/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA sin.
        class SinCudaBuiltIn : public concepts::Implements<ConceptMathSin, SinCudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto sin(
                    SinCudaBuiltIn const & sin_ctx,
                    TArg const & arg)
                -> decltype(::sin(arg))
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sin(arg);
                }
            };
            //! The CUDA sin float specialization.
            template<>
            struct Sin<
                SinCudaBuiltIn,
                float>
            {
                __device__ static auto sin(
                    SinCudaBuiltIn const & sin_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sinf(arg);
                }
            };
        }
    }
}

#endif
