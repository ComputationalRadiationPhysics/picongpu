/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
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

#include <alpaka/math/exp/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in exp.
        class ExpCudaBuiltIn
        {
        public:
            using ExpBase = ExpCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA exp trait specialization.
            template<
                typename TArg>
            struct Exp<
                ExpCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto exp(
                    ExpCudaBuiltIn const & exp_ctx,
                    TArg const & arg)
                -> decltype(::exp(arg))
                {
                    alpaka::ignore_unused(exp_ctx);
                    return ::exp(arg);
                }
            };
            //! The CUDA exp float specialization.
            template<>
            struct Exp<
                ExpCudaBuiltIn,
                float>
            {
                __device__ static auto exp(
                    ExpCudaBuiltIn const & exp_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(exp_ctx);
                    return ::expf(arg);
                }
            };
        }
    }
}

#endif
